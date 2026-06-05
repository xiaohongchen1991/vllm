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

    fp8_min, fp8_max = get_fp8_min_max()
    eps = 1e-10

    inputs = {}

    for hidden_size, group_size, num_tokens in product(
        hidden_size_list, group_size_list, num_tokens_list
    ):
        input = torch.randn(num_tokens, hidden_size, device="cuda", dtype=in_dtype) * 8
        output_q = torch.empty(input.shape, device=input.device, dtype=out_dtype)
        num_groups_per_row = hidden_size // group_size
        k_num_packed = (num_groups_per_row + 3) // 4
        tma_aligned_num_tokens = ((num_tokens + 3) // 4) * 4
        output_s_packed = torch.empty_strided(
            (num_tokens, k_num_packed),
            (1, tma_aligned_num_tokens),
            device=input.device,
            dtype=scale_dtype,
        )
        config_key = CaseKey(
            {
                "hidden_size": hidden_size,
                "group_size": group_size,
                "num_tokens": num_tokens,
            }
        )
        inputs[config_key] = (
            input,
            output_q,
            output_s_packed,
            group_size,
            eps,
            fp8_min,
            fp8_max,
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
      3. Among the num_tokens values tuned for that hidden_size and group_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.
    """

    if not config_keys:
        return None

    input, _, _, group_size, *_ = args
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
    input: torch.Tensor,  # [num_tokens, hidden_size]
    output_q: torch.Tensor,  # [output_q_num_tokens, hidden_size]
    output_s_packed: torch.Tensor,  # [num_tokens, padded_groups_per_row]
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
) -> None:
    return


def baseline(
    input: torch.Tensor,  # [num_tokens, hidden_size]
    output_q: torch.Tensor,  # [output_q_num_tokens, hidden_size]
    output_s_packed: torch.Tensor,  # [num_tokens, padded_groups_per_row]
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
) -> None:
    num_tokens, hidden_size = input.shape
    tma_aligned_num_tokens = ((num_tokens + 3) // 4) * 4
    groups_per_row = hidden_size // group_size
    k_num_packed = (groups_per_row + 3) // 4
    num_scale_elems = num_tokens + (k_num_packed - 1) * tma_aligned_num_tokens
    output_q_num_tokens = output_q.shape[0]

    if tma_aligned_num_tokens > num_tokens and k_num_packed > 1:
        torch.as_strided(
            output_s_packed,
            size=(k_num_packed - 1, tma_aligned_num_tokens - num_tokens),
            stride=(tma_aligned_num_tokens, 1),
            storage_offset=num_tokens,
        ).zero_()

    if output_q_num_tokens > num_tokens:
        output_q[num_tokens:output_q_num_tokens, :].zero_()

    torch.ops._C.per_token_group_fp8_quant_packed(
        input,
        output_q,
        output_s_packed,
        group_size,
        eps,
        min_8bit,
        max_8bit,
    )

@register_kernel(
    mutates_args=["output_q", "output_s_packed"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def per_token_group_fp8_quant_packed(
    input: torch.Tensor,  # [num_tokens, hidden_size]
    output_q: torch.Tensor,  # [output_q_num_tokens, hidden_size]
    output_s_packed: torch.Tensor,  # [num_tokens, padded_groups_per_row]
    group_size: int,
    eps: float,
    min_8bit: float,
    max_8bit: float,
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)

    assert group_size == 128
    assert input.dtype in (torch.float16, torch.bfloat16)

    assert input.is_contiguous()
    assert output_q.is_contiguous()

    assert hidden_size % group_size == 0
    groups_per_row = hidden_size // group_size

    k_num_packed = output_s_packed.shape[-1]
    tma_aligned_num_tokens = output_s_packed.stride(1)
    padded_groups_per_row = k_num_packed * 4
    hl.specialize(k_num_packed)
    hl.specialize(padded_groups_per_row)

    assert output_s_packed.shape == (num_tokens, (groups_per_row + 3) // 4)
    assert output_s_packed.stride() == (1, ((num_tokens + 3) // 4) * 4)
    assert output_s_packed.dtype == torch.int32

    output_q_num_tokens = output_q.shape[0]
    assert output_q.shape == (output_q_num_tokens, hidden_size)
    assert output_q_num_tokens >= num_tokens

    # zero out padding part
    num_scale_elems = num_tokens + (k_num_packed - 1) * tma_aligned_num_tokens
    if tma_aligned_num_tokens > num_tokens and k_num_packed > 1:
        torch.as_strided(
            output_s_packed,
            size=(k_num_packed - 1, tma_aligned_num_tokens - num_tokens),
            stride=(tma_aligned_num_tokens, 1),
            storage_offset=num_tokens,
        ).zero_()

    if output_q_num_tokens > num_tokens:
        output_q[num_tokens:output_q_num_tokens, :].zero_()

    input = input.view(num_tokens, -1, group_size)
    output_q = output_q.view(output_q_num_tokens, -1, group_size)

    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, k_num_packed, group_size],
        block_size=[1, None, group_size]
    ):
        packed_s_blk = hl.zeros([tile_m, tile_gn], dtype=torch.int32)
        for i in hl.static_range(4):
            tile_g = tile_gn.index * 4 + i
            mask_g = tile_g < groups_per_row
            x_blk = hl.load(input, [tile_m, tile_g, tile_n], extra_mask=mask_g[None, :, None]).to(torch.float32)

            y_s_blk = torch.clamp(torch.amax(torch.abs(x_blk), dim=-1), min=eps)
            y_s_blk = y_s_blk / max_8bit
            y_s_blk = torch.clamp(y_s_blk, min=1.0e-10)

            exp_unbiased_blk = torch.ceil(torch.log2(y_s_blk))
            exp_byte_blk = exp_unbiased_blk.to(torch.int32) + 127
            exp_byte_blk = torch.where(
                mask_g[None, :],
                exp_byte_blk,
                0
            )
            packed_s_blk = packed_s_blk | exp_byte_blk << (i * 8)

            y_s_blk = torch.exp2(exp_unbiased_blk.to(torch.float32))
            y_q_blk = torch.clamp(x_blk / y_s_blk[:, :, None], min_8bit, max_8bit).to(
                output_q.dtype
            )

            hl.store(output_q, [tile_m, tile_g, tile_n], y_q_blk, extra_mask=mask_g[None, :, None])

        output_s_packed[tile_m, tile_gn] = packed_s_blk
