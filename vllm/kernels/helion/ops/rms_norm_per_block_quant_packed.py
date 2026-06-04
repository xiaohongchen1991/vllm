# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import product
from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
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


def _get_fp8_dtype() -> torch.dtype:
    return current_platform.fp8_dtype()


def _get_int8_min_max() -> tuple[int, int]:
    qtype_traits = torch.iinfo(torch.int8)
    return qtype_traits.min, qtype_traits.max


def _get_int8_min_scaling_factor() -> float:
    return torch.finfo(torch.float32).eps


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover all
    # input property combination. Currently, dtypes are fixed. We need
    # optimization to bucket/skip some combinations
    num_tokens_list = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    hidden_size_list = [2048, 4096, 5120]
    group_size_list = [128]
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.int32
    inputs = {}

    for hidden_size, group_size, num_tokens in product(
        hidden_size_list, group_size_list, num_tokens_list
    ):
        input = torch.randn(num_tokens, hidden_size, device="cuda", dtype=in_dtype) * 8.0
        result = torch.empty(input.shape, device=input.device, dtype=out_dtype)
        groups_per_row = hidden_size // group_size
        packed_groups_per_row = (groups_per_row + 3) // 4
        tma_aligned_num_tokens = ((num_tokens + 3) // 4) * 4
        scale = torch.empty_strided(
            (num_tokens, packed_groups_per_row),
            (1, tma_aligned_num_tokens),
            device=input.device,
            dtype=scale_dtype,
        )
        residual = torch.randn_like(input)
        weight = torch.normal(
            mean=1.0,
            std=1.0,
            size=(hidden_size,),
            dtype=input.dtype,
            device=input.device,
        )
        epsilon = 1e-6

        config_key = CaseKey(
            {
                "hidden_size": hidden_size,
                "group_size": group_size,
                "num_tokens": num_tokens,
            }
        )
        inputs[config_key] = (
            result,
            input,
            weight,
            scale,
            epsilon,
            residual,
            group_size,
        )

    return inputs


_pick_cache: dict[tuple[int, int, int], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest hidden_size among available configs
         (exact match preferred).
      2. Find the closest group_size among available configs
         (exact match preferred).
      2. Among the num_tokens values tuned for that hidden_size and group_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.
    """

    if not config_keys:
        return None

    _, input, _, _, _, _, group_size = args
    num_tokens, hidden_size = input.shape

    cache_key = (num_tokens, group_size, hidden_size)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    configs: dict[int, dict[int, list[int]]] = {}
    for key in config_keys:
        if key.is_default():
            continue
        configs.setdefault(key["hidden_size"], {}).setdefault(
            key["group_size"], []
        ).append(key["num_tokens"])

    if not configs:
        return None

    best_hidden_size = min(configs, key=lambda s: abs(s - hidden_size))
    best_group_size = min(configs[best_hidden_size], key=lambda s: abs(s - group_size))
    available_num_tokens = sorted(configs[best_hidden_size][best_group_size])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens), available_num_tokens[-1]
    )

    result = CaseKey(
        {
            "hidden_size": best_hidden_size,
            "group_size": best_group_size,
            "num_tokens": best_num_tokens,
        }
    )
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    weight: torch.Tensor,  # [hidden_size]
    scale: torch.Tensor,  # [num_tokens, packed_groups_per_row]
    epsilon: float,
    residual: torch.Tensor | None,  # [num_tokens, hidden_size]
    group_size: int,
) -> None:
    return


def baseline(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    weight: torch.Tensor,  # [hidden_size]
    scale: torch.Tensor,  # [num_tokens, packed_groups_per_row]
    epsilon: float,
    residual: torch.Tensor | None,  # [num_tokens, hidden_size]
    group_size: int,
) -> None:
    num_tokens, hidden_size = input.shape
    tma_aligned_num_tokens = ((num_tokens + 3) // 4) * 4
    groups_per_row = hidden_size // group_size
    packed_groups_per_row = (groups_per_row + 3) // 4

    if tma_aligned_num_tokens > num_tokens and packed_groups_per_row > 1:
        torch.as_strided(
            scale,
            size=(packed_groups_per_row - 1, tma_aligned_num_tokens - num_tokens),
            stride=(tma_aligned_num_tokens, 1),
            storage_offset=num_tokens,
        ).zero_()

    rms = torch.empty_like(input)
    rms.copy_(input)
    if residual is not None:
        torch.ops._C.fused_add_rms_norm(rms, residual, weight, epsilon)
    else:
        torch.ops._C.rms_norm(rms, input, weight, epsilon)

    qtype_traits_min, qtype_traits_max = get_fp8_min_max()
    torch.ops._C.per_token_group_fp8_quant_packed(
        rms,
        result,
        scale,
        group_size,
        1.0e-10,
        qtype_traits_min,
        qtype_traits_max,
    )

@register_kernel(
    mutates_args=["result", "scale", "residual"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
        autotune_config_overrides={"static_ranges": []},
    ),
)  # type: ignore[misc]
def rms_norm_per_block_quant_packed(
    result: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    weight: torch.Tensor,  # [hidden_size]
    scale: torch.Tensor,  # [num_tokens, packed_groups_per_row]
    epsilon: float,
    residual: torch.Tensor | None,  # [num_tokens, hidden_size]
    group_size: int,
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    assert input.dtype in (torch.float16, torch.bfloat16)
    assert group_size == 128

    hl.specialize(hidden_size)
    hl.specialize(group_size)

    assert hidden_size % group_size == 0
    groups_per_row = hidden_size // group_size
    hl.specialize(groups_per_row)

    packed_groups_per_row = scale.shape[-1]
    tma_aligned_num_tokens = scale.stride(1)
    padded_groups_per_row = packed_groups_per_row * 4
    hl.specialize(packed_groups_per_row)
    hl.specialize(padded_groups_per_row)

    assert scale.shape == (num_tokens, (groups_per_row + 3) // 4)
    assert scale.stride() == (1, ((num_tokens + 3) // 4) * 4)
    assert scale.dtype == torch.int32

    fp8_dtype = _get_fp8_dtype()
    assert result.dtype == fp8_dtype
    assert result.is_contiguous() and input.is_contiguous()

    assert input.dtype == weight.dtype

    if residual is not None:
        assert residual.dtype == input.dtype

    qtype_traits_min, qtype_traits_max = get_fp8_min_max()
    min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)

    # zero out padding
    if tma_aligned_num_tokens > num_tokens and packed_groups_per_row > 1:
        torch.as_strided(
            scale,
            size=(packed_groups_per_row - 1, tma_aligned_num_tokens - num_tokens),
            stride=(tma_aligned_num_tokens, 1),
            storage_offset=num_tokens,
        ).zero_()

    input_3d = input.view(num_tokens, -1, group_size)
    result_3d = result.view(num_tokens, -1, group_size)
    weight_2d = weight.view(-1, group_size)
    if residual is not None:
        residual_3d = residual.view(num_tokens, -1, group_size)

    for tile_m in hl.tile(num_tokens, block_size=1):
        rms = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
            rms = rms + x_blk.pow(2).sum(dim=-1)

        rms = torch.rsqrt(rms * (1.0 / hidden_size) + epsilon)

        for tile_gn, tile_n in hl.tile(
            [packed_groups_per_row, group_size], block_size=[None, group_size]
        ):
            packed_s_blk = hl.zeros([tile_m, tile_gn], dtype=torch.int32)
            for i in hl.static_range(4):
                tile_g = tile_gn.index * 4 + i
                mask_g = tile_g < groups_per_row

                x_blk = hl.load(input_3d, [tile_m, tile_g, tile_n], extra_mask=mask_g[None, :, None])
                if residual is not None:
                    r_blk = hl.load(residual_3d, [tile_m, tile_g, tile_n], extra_mask=mask_g[None, :, None])
                    x_blk = x_blk + r_blk

                x_blk = x_blk.to(torch.float32)
                w_blk = hl.load(weight_2d, [tile_g, tile_n], extra_mask=mask_g[:, None])
                x_norm_blk = (x_blk * rms[:, None, None]).to(input.dtype) * w_blk
                s_blk = torch.amax(torch.abs(x_norm_blk), dim=-1).to(torch.float32)
                s_blk = s_blk / qtype_traits_max
                s_blk = s_blk.clamp(min=min_scaling_factor)

                s_exp_blk = torch.ceil(torch.log2(s_blk))
                s_byte_blk = s_exp_blk.to(torch.int32) + 127
                s_byte_blk = torch.where(
                    mask_g[None, :],
                    s_byte_blk,
                    0
                )
                packed_s_blk = packed_s_blk | (s_byte_blk << (i * 8))

                s_blk = torch.exp2(s_exp_blk)
                y_blk = x_norm_blk / s_blk[:, :, None]
                y_blk = y_blk.clamp(qtype_traits_min, qtype_traits_max).to(result.dtype)
                hl.store(result_3d, [tile_m, tile_g, tile_n], y_blk, extra_mask=mask_g[None, :, None])

                # store residual after all uses of the old value are finished
                if residual is not None:
                    hl.store(
                        residual_3d, [tile_m, tile_g, tile_n], x_blk.to(residual.dtype), extra_mask=mask_g[None, :, None]
                    )

            scale[tile_m, tile_gn] = packed_s_blk

