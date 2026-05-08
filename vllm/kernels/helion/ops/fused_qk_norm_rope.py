# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from itertools import product
from typing import Any

import regex as re
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
import helion.language as hl

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)

def _compute_cos_sin_cache(max_position_embeddings, rotary_dim, device="cuda", dtype=torch.float):

    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim
        )
    )

    t = torch.arange(max_position_embeddings, device=device, dtype=dtype)

    freqs = torch.einsum("i,j -> ij", t, inv_freq)
    cos = freqs.cos()
    sin = freqs.sin()
    cache = torch.cat((cos, sin), dim=-1)
    return cache

def generate_inputs() -> dict[str, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover
    # all input property combination. Currently, dtypes are fixed. We need
    # optimization to bucket/skip some combinations
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    # num_tokens_list = [32]
    num_heads_pair = [
        # Qwen3-1.7B
        (16, 8),

        # Qwen3-8B
        (32, 8),

        # Qwen3-32B
        (64, 8),
    ]
    head_dim = 128
    in_dtype: torch.dtype = torch.bfloat16
    rotary_ratio = 1.0
    is_neox = True
    eps = 1e-6
    device="cuda"
    inputs = {}

    for num_tokens, (num_q_heads, num_kv_heads) in product(
        num_tokens_list, num_heads_pair
    ):
        total_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
        qkv = torch.randn(num_tokens, total_dim, dtype=in_dtype, device=device)
        positions = torch.arange(num_tokens, dtype=torch.long, device=device)
        q_weight = torch.normal(
            mean=1.0,
            std=1.0,
            size=(head_dim,),
            dtype=qkv.dtype,
            device=device,
        )
        k_weight = torch.normal(
            mean=1.0,
            std=1.0,
            size=(head_dim,),
            dtype=qkv.dtype,
            device=device,
        )
        rotary_dim = int(head_dim * rotary_ratio)
        cos_sin_cache = _compute_cos_sin_cache(40960, rotary_dim)
        cos_sin_cache = cos_sin_cache.to(in_dtype)

        config_key = (
            f"q_heads_{num_q_heads}_"
            f"kv_heads_{num_kv_heads}_num_tokens_{num_tokens}"
        )
        inputs[config_key] = (
            qkv,
            num_q_heads,
            num_kv_heads,
            num_kv_heads,
            head_dim,
            eps,
            q_weight,
            k_weight,
            cos_sin_cache,
            is_neox,
            positions.view(-1)
        )

    return inputs
        
def pick_config(args: tuple[Any, ...], config_keys: list[str]) -> str | None:
    if not config_keys:
        return None

    qkv, q_heads, kv_heads, *_ = args
    num_tokens = qkv.shape[0]

    configs: dict[int, dict[int, list[int]]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(
            r"q_heads_(\d+)_kv_heads_(\d+)_num_tokens_(\d+)", key
        )
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format 'q_heads_{{int}}_"
                f"kv_heads_{{int}}_num_tokens_{{int}}'"
            )
        q_heads_str, kv_heads_str, num_tokens_str = match.groups()
        configs.setdefault(int(q_heads_str), {}).setdefault(
            int(kv_heads_str), []
        ).append(int(num_tokens_str))

    if not configs:
        return "default" if "default" in config_keys else None

    best_q_heads = min(configs, key=lambda s: abs(s - q_heads))
    best_kv_heads = min(
        configs[best_q_heads], key=lambda s: abs(s - kv_heads)
    )
    available_num_tokens = sorted(configs[best_q_heads][best_kv_heads])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens), available_num_tokens[-1]
    )

    return (
        f"q_heads_{best_q_heads}_kv_heads_"
        f"{best_kv_heads}_num_tokens_{best_num_tokens}"
    )

def fake_impl(
    qkv: torch.Tensor, # [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor, # [max_position, rotary_dim]
    is_neox: bool,
    position_ids: torch.Tensor, # [num_tokens],
    forced_token_heads_per_warp: int = -1, # dummy
) -> None:
    return

@register_kernel(
    mutates_args=["qkv"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
# @helion.kernel(
#     autotune_effort="none",
#     static_shapes=False,
#     ignore_warnings=[helion.exc.TensorOperationInWrapper],
# )
def fused_qk_norm_rope(
    qkv: torch.Tensor, # [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor, # [max_position, rotary_dim]
    is_neox: bool,
    position_ids: torch.Tensor, # [num_tokens],
    forced_token_heads_per_warp: int = -1, # dummy
) -> None:
    assert qkv.ndim == 2
    num_tokens = qkv.shape[0]
    total_heads = num_heads_q + num_heads_k + num_heads_v
    assert qkv.shape[1] == total_heads * head_dim
    hl.specialize(qkv.shape[1])

    assert cos_sin_cache.ndim == 2
    max_position, rotary_dim = cos_sin_cache.shape
    hl.specialize(max_position)
    hl.specialize(rotary_dim)
    assert rotary_dim % 2 == 0
    assert rotary_dim <= head_dim
    embed_dim = rotary_dim // 2;

    hl.specialize(num_heads_q)
    hl.specialize(num_heads_k)
    hl.specialize(num_heads_v)
    hl.specialize(head_dim)
    hl.constexpr(is_neox)

    assert position_ids.ndim == 1 and position_ids.shape[0] == num_tokens
    hl.specialize(position_ids.shape[0])

    assert q_weight.ndim == 1 and q_weight.shape[0] == head_dim
    hl.specialize(q_weight.shape[0])
    assert k_weight.ndim == 1 and k_weight.shape[0] == head_dim
    hl.specialize(k_weight.shape[0])

    assert qkv.dtype == q_weight.dtype and q_weight.dtype == k_weight.dtype
    assert position_ids.dtype == torch.int64

    assert qkv.is_contiguous()
    assert position_ids.is_contiguous()
    assert q_weight.is_contiguous()
    assert k_weight.is_contiguous()
    assert cos_sin_cache.is_contiguous()

    qk_heads = num_heads_q + num_heads_k;

    qkv = qkv.view(num_tokens, -1, head_dim)

    for tile_m, tile_gn, tile_n in hl.tile([num_tokens, qk_heads, head_dim], block_size=[1, None, head_dim]):
        x_blk = qkv[tile_m, tile_gn, tile_n].to(dtype=torch.float32)

        # Helion will introduce a reduction block_size for sum along last dim.
        # Need to explicitly add tile_n with head_dim block_size in the above
        # tile loop to help helion infer the right reduction block_size
        rms = x_blk.pow(2).sum(dim=-1)
        rms = torch.rsqrt(rms * (1.0 / head_dim) + eps)

        use_q_weight = (tile_gn.index < num_heads_q)[None, :, None]
        w_blk = torch.where(
            use_q_weight,
            q_weight[None, None, tile_n],
            k_weight[None, None, tile_n]
        )

        x_blk = (x_blk * rms[:, :, None]).to(qkv.dtype) * w_blk

        qkv[tile_m, tile_gn, tile_n] = x_blk
            
        pos_id = position_ids[tile_m]
        cos_blk = cos_sin_cache[pos_id, hl.arange(embed_dim)]
        sin_blk = cos_sin_cache[pos_id, hl.arange(embed_dim) + embed_dim]

        if is_neox:
            x1_offset = hl.arange(embed_dim)
            x2_offset = x1_offset + embed_dim
        else:
            x1_offset =	hl.arange(embed_dim) * 2
            x2_offset = x1_offset + 1

        x1_blk = qkv[tile_m, tile_gn, x1_offset]
        x2_blk = qkv[tile_m, tile_gn, x2_offset]

        o1_blk = x1_blk * cos_blk[:, None, :] - x2_blk * sin_blk[:, None, :]
        o2_blk = x2_blk * cos_blk[:, None, :] + x1_blk * sin_blk[:, None, :]

        qkv[tile_m, tile_gn, x1_offset] = o1_blk
        qkv[tile_m, tile_gn, x2_offset] = o2_blk


import torch.nn as nn
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.config import VllmConfig, set_current_vllm_config

class Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        qkv: torch.Tensor, # [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
        num_heads_q: int,
        num_heads_k: int,
        num_heads_v: int,
        head_dim: int,
        eps: float,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin_cache: torch.Tensor, # [max_position, rotary_dim]
        is_neox: bool,
        position_ids: torch.Tensor, # [num_tokens]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        q_size = num_heads_q * head_dim
        kv_size = num_heads_k * head_dim

        q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
        q_by_head = RMSNorm.forward_static(q_by_head, eps, head_dim, qkv.dtype, q_weight)
        q = q_by_head.view(q.shape)
        
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
        k_by_head = RMSNorm.forward_static(k_by_head, eps, head_dim, qkv.dtype, k_weight)
        k = k_by_head.view(k.shape)

        q, k =  RotaryEmbedding.forward_static(position_ids, q, k, head_dim, cos_sin_cache.shape[1], cos_sin_cache, is_neox)
        return q, k, v
        

config = VllmConfig()
with set_current_vllm_config(config):
    layer = Layer()
    compiled_layer = torch.compile(layer.forward)


def baseline(
    qkv: torch.Tensor, # [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor, # [max_position, rotary_dim]
    is_neox: bool,
    position_ids: torch.Tensor, # [num_tokens],
    forced_token_heads_per_warp: int = -1, # dummy
):
    return compiled_layer(qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps, q_weight, k_weight, cos_sin_cache, is_neox, position_ids)
    # torch.ops._C.fused_qk_norm_rope(qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps, q_weight, k_weight, cos_sin_cache, is_neox, position_ids)
    

def helion_kernel(
    qkv: torch.Tensor, # [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor, # [max_position, rotary_dim]
    is_neox: bool,
    position_ids: torch.Tensor, # [num_tokens],
    forced_token_heads_per_warp: int = -1, # dummy
):
    fused_qk_norm_rope(qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps, q_weight, k_weight, cos_sin_cache, is_neox, position_ids, forced_token_heads_per_warp)
    return qkv.split([num_heads_q*head_dim, num_heads_k*head_dim, num_heads_v*head_dim], dim=-1)

# from vllm.model_executor.layers.layernorm import RMSNorm
# from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
# from vllm.config import VllmConfig, set_current_vllm_config
# from vllm.utils.torch_utils import set_random_seed

# def _apply_qk_norm_rope(
#     qkv: torch.Tensor,
#     positions: torch.Tensor,
#     q_norm: RMSNorm,
#     k_norm: RMSNorm,
#     rope: RotaryEmbedding,
#     num_heads_q: int,
#     num_heads_kv: int,
#     head_dim: int,
# ) -> torch.Tensor:
#     q_size = num_heads_q * head_dim
#     kv_size = num_heads_kv * head_dim

#     q, k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)

#     q_by_head = q.view(*q.shape[:-1], q.shape[-1] // head_dim, head_dim)
#     q_by_head = q_norm.forward_native(q_by_head)
#     q = q_by_head.view(q.shape)

#     k_by_head = k.view(*k.shape[:-1], k.shape[-1] // head_dim, head_dim)
#     k_by_head = k_norm.forward_native(k_by_head)
#     k = k_by_head.view(k.shape)

#     q, k = rope.forward_native(positions, q, k)
#     return torch.cat([q, k, v], dim=-1)


# @torch.inference_mode()
# def test():
#     device = "cuda"
#     torch.set_default_device(device)
#     set_random_seed(13)
#     num_heads, num_kv_heads, head_dim = 32, 8, 128
#     num_tokens = 32
#     dtype = torch.bfloat16
#     rotary_ratio = 1.0
#     is_neox = True
#     eps = 1e-5

#     total_dim = (num_heads + 2 * num_kv_heads) * head_dim
#     qkv_base = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
#     qkv_fused = qkv_base.clone()
#     positions = torch.arange(num_tokens, dtype=torch.long, device=device)

#     q_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
#     k_norm = RMSNorm(head_dim, eps=eps).to(device=device, dtype=dtype)
#     q_norm.weight.data.normal_(mean=1.0, std=0.1)
#     k_norm.weight.data.normal_(mean=1.0, std=0.1)
#     q_weight = q_norm.weight.data
#     k_weight = k_norm.weight.data
#     rotary_dim = int(head_dim * rotary_ratio)
#     rope = RotaryEmbedding(
#         head_size=head_dim,
#         rotary_dim=rotary_dim,
#         max_position_embeddings=4096,
#         base=10000.0,
#         is_neox_style=is_neox,
#         dtype=dtype,
#     ).to(device)

#     ref_result = _apply_qk_norm_rope(
#         qkv=qkv_base,
#         positions=positions,
#         q_norm=q_norm,
#         k_norm=k_norm,
#         rope=rope,
#         num_heads_q=num_heads,
#         num_heads_kv=num_kv_heads,
#         head_dim=head_dim,
#     )

#     fused_qk_norm_rope(
#         qkv_fused,
#         num_heads,
#         num_kv_heads,
#         num_kv_heads,
#         head_dim,
#         eps,
#         q_weight,
#         k_weight,
#         rope.cos_sin_cache,
#         is_neox,
#         positions.view(-1)
#     )

#     torch.testing.assert_close(
#         qkv_fused,
#         ref_result,
#         atol=1e-2,
#         rtol=1e-2,
#     )

# if __name__ == "__main__":
#     config = VllmConfig()
#     with set_current_vllm_config(config):
#         test()
