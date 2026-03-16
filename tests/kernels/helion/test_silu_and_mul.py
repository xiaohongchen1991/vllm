# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rms_norm helion kernel

Run `pytest tests/kernels/helion/test_silu_and_mul.py`.
"""

import pytest
import torch

from vllm.kernels.helion.ops.silu_and_mul import (
    silu_and_mul,
)
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 13824]  # Arbitrary values for testing
SEEDS = [0]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_act_and_mul(
    default_vllm_config,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    set_random_seed(seed)
    torch.set_default_device("cuda")
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)

    ref_out = torch.empty((num_tokens, d), device=x.device, dtype=dtype)
    torch.ops._C.silu_and_mul(ref_out, x)

    ops_out = torch.empty((num_tokens, d), device=x.device, dtype=dtype)
    silu_and_mul(ops_out, x)

    torch.testing.assert_close(ops_out, ref_out)
