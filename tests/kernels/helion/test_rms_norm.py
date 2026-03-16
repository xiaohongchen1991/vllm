# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rms_norm helion kernel

Run `pytest tests/kernels/helion/test_rms_norm.py`.
"""

import pytest
import torch

from vllm.kernels.helion.ops.rms_norm import (
    rms_norm,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 4096]  # Arbitrary values for testing
HIDDEN_SIZES = [8, 768, 769, 5120, 5125, 8192]  # Arbitrary values for testing
ADD_RESIDUAL = [False, True]
SEEDS = [0]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("strided_input", [False, True])
@torch.inference_mode()
def test_rms_norm(
    default_vllm_config,
    num_tokens: int,
    hidden_size: int,
    add_residual: bool,
    dtype: torch.dtype,
    seed: int,
    strided_input: bool,
) -> None:
    set_random_seed(seed)
    torch.set_default_device("cuda")
    layer = RMSNorm(hidden_size).to(dtype=dtype)
    layer.weight.data.normal_(mean=1.0, std=0.1)
    scale = 1 / (2 * hidden_size)
    last_dim = 2 * hidden_size if strided_input else hidden_size
    x = torch.randn(num_tokens, last_dim, dtype=dtype)
    x = x[..., :hidden_size]
    assert x.is_contiguous() != strided_input
    x *= scale
    residual = torch.randn_like(x) * scale if add_residual else None

    if add_residual:
        ref_out, ref_residual = layer.forward_native(x, residual)
    else:
        ref_out = layer.forward_native(x, residual)

    ops_out = torch.empty(x.shape, device=x.device, dtype=dtype)
    ops_residual = residual.clone() if residual is not None else None

    rms_norm(ops_out, x, layer.weight, layer.variance_epsilon, ops_residual)

    # LayerNorm operators (including RMS) typically have larger
    # numerical errors than other operators because they involve reductions.
    # Therefore, we use a larger tolerance.
    torch.testing.assert_close(ops_out, ref_out, atol=1e-2, rtol=1e-2)
    if add_residual:
        torch.testing.assert_close(ops_residual, ref_residual, atol=1e-2, rtol=1e-2)
