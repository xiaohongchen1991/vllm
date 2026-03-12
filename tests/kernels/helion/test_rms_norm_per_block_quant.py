# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rms_norm_per_block_quant helion kernel

Run `pytest tests/kernels/helion/test_rms_norm_per_block_quant.py`.
"""
import itertools

import pytest
import torch

import vllm._custom_ops as ops
from vllm.kernels.helion.ops.rms_norm_per_block_quant import (
    rms_norm_per_block_quant,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

DTYPES = [torch.bfloat16, torch.float]
QUANT_DTYPES = [torch.int8, current_platform.fp8_dtype()]
VEC_HIDDEN_SIZES = [64, 1024]
# Avoid combinatorial explosion with full Cartesian product
NUM_TOKENS_HIDDEN_SIZES = [
    *[(1, i) for i in [64, 128, 1024, 5120]],
    *[(2048, i) for i in [64, 1024]],
    *[(4096, i) for i in [64]],
]

ADD_RESIDUAL = [False, True]
SCALE_UBS = [True, False]
GROUP_SIZES = [[1, 64], [1, 128]]
TMA_ALIGNMENTS = [0, 4]

SEEDS = [0]

EPS = 1e-6


@pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("has_scale_ub", SCALE_UBS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
@pytest.mark.parametrize("is_scale_transposed", [False, True])
@pytest.mark.parametrize(
    "group_size, tma_alignment",
    [*itertools.product(GROUP_SIZES, TMA_ALIGNMENTS)],
)
@pytest.mark.parametrize("seed", SEEDS)
def test_rms_norm(
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    has_scale_ub: bool,
    dtype: torch.dtype,
    quant_dtype: torch.dtype,
    is_scale_transposed: bool,
    group_size: list[int] | None,
    tma_alignment: int,
    seed: int,
) -> None:
    set_random_seed(seed)

    if hidden_size % group_size[1] != 0:
        # skip
        return

    if (
        tma_alignment != 0
        and hidden_size // group_size[1] % tma_alignment == 0
    ):
        # Skip tests where TMA alignment doesn't create extra padding to save time
        return

    if has_scale_ub and quant_dtype != current_platform.fp8_dtype():
        # skip
        return

    scale = 1 / (hidden_size)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * scale
    weight = torch.normal(
        mean=1.0, std=1.0, size=(hidden_size,), dtype=dtype, device=x.device
    )
    residual = torch.randn_like(x) * scale if add_residual else None
    scale_ub = (
        torch.mean(x).to(dtype=torch.float32, device="cuda") if has_scale_ub else None
    )
    groups_per_row = hidden_size // group_size[1]

    ref_residual = residual.clone() if residual is not None else None
    ref_out, ref_scales = ops.rms_norm_per_block_quant(
        x, weight, EPS, quant_dtype, group_size, scale_ub, ref_residual, is_scale_transposed, tma_alignment
    )
    ref_scales = ref_scales.contiguous()

    ops_out = torch.empty(x.shape, device=x.device, dtype=quant_dtype)
    if is_scale_transposed:
        if tma_alignment == 0:
            ops_scales = torch.empty((groups_per_row, num_tokens), device=x.device, dtype=torch.float32).transpose(0, 1)
        else:
            tma_aligned_m = (num_tokens + tma_alignment - 1) // tma_alignment * tma_alignment
            shape = (num_tokens, groups_per_row)
            stride = (1, tma_aligned_m)
            ops_scales = torch.empty_strided(
                shape, stride, device=x.device, dtype=torch.float32
            )
    else:
        ops_scales = torch.empty((num_tokens, groups_per_row), device=x.device, dtype=torch.float32)

    ops_residual = residual.clone() if residual is not None else None
    rms_norm_per_block_quant(
        ops_out, x, weight, ops_scales, EPS, scale_ub, ops_residual, group_size[1], is_scale_transposed
    )
    ops_scales = ops_scales.contiguous()

    torch.testing.assert_close(ref_scales, ops_scales)
    # allow 1 ULP difference
    assert (
        ref_out.view(torch.uint8).to(torch.int16)
        - ops_out.view(torch.uint8).to(torch.int16)
    ).abs().max() <= 1

    if add_residual:
        torch.testing.assert_close(ref_residual, ops_residual)
