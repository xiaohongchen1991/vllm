# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rms_norm helion kernel
Run `pytest tests/kernels/helion/test_silu_and_mul_per_block_quant.py`.
"""

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.kernels.helion.utils import skip_if_platform_unsupported
from vllm.kernels.helion.config_manager import ConfigManager

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.kernels.helion.ops.silu_and_mul_per_block_quant import (
    silu_and_mul_per_block_quant,
    baseline
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )

@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestSiluAndMulPerBlockQuantCorrectness:
    @pytest.mark.parametrize("num_tokens", [1, 7, 4096])
    @pytest.mark.parametrize("hidden_size", [1024, 2048, 5120])
    @pytest.mark.parametrize("group_size", [64, 128])
    @pytest.mark.parametrize("is_scale_transposed", [False, True])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("quant_dtype", [current_platform.fp8_dtype(), torch.int8])
    @pytest.mark.parametrize("seed", [0])
    def test_silu_and_mul_per_block_quant(
        self,
        num_tokens: int,
        hidden_size: int,
        group_size: int,
        is_scale_transposed: bool,
        dtype: torch.dtype,
        quant_dtype: torch.dtype,
        seed: int,
    ) -> None:
        skip_if_platform_unsupported("silu_and_mul_per_block_quant")
        set_random_seed(seed)

        if hidden_size % group_size != 0:
            return

        scale = 1 / hidden_size
        x = torch.randn(num_tokens, 2 * hidden_size, dtype=dtype, device="cuda") * scale

        ref_out = torch.empty(num_tokens, hidden_size, device="cuda", dtype=quant_dtype)
        ref_scales = torch.empty((x.shape[0], hidden_size // group_size), device="cuda", dtype=torch.float32)
        baseline(ref_out, x, ref_scales, group_size, None, False)

        ops_out = torch.empty(num_tokens, hidden_size, device="cuda", dtype=quant_dtype)
        ops_scales = torch.empty((x.shape[0], hidden_size // group_size), device="cuda", dtype=torch.float32)
        silu_and_mul_per_block_quant(ops_out, x, ops_scales, group_size, None, False)

        torch.testing.assert_close(ref_scales, ops_scales)
        # allow 1 ULP difference
        assert (
            ref_out.view(torch.uint8).to(torch.int16)
            - ops_out.view(torch.uint8).to(torch.int16)
        ).abs().max() <= 1
