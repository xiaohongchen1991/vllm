# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rms_norm helion kernel
Run `pytest tests/kernels/helion/test_silu_and_mul_dynamic_per_token_quant.py`.
"""

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.config_manager import ConfigManager

from vllm.model_executor.layers.activation import SiluAndMul
from vllm.kernels.helion.ops.silu_and_mul_dynamic_per_token_quant import (
    silu_and_mul_dynamic_per_token_quant,
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


class TestSiluAndMulDynamicPerTokenQuantCorrectness:
    @pytest.mark.parametrize("num_tokens", [1, 7, 4096])
    @pytest.mark.parametrize("intermediate_size", [17, 1024, 1025, 1026, 5137, 8193])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float])
    @pytest.mark.parametrize("has_scale_ub", [True, False])
    @pytest.mark.parametrize("seed", [0])
    def test_silu_and_mul_dynamic_per_token_quant(
        self,
        num_tokens: int,
        intermediate_size: int,
        dtype: torch.dtype,
        has_scale_ub: bool,
        seed: int,
    ) -> None:
        skip_if_platform_unsupported("silu_and_mul_dynamic_per_token_quant")
        set_random_seed(seed)

        x = (
            torch.rand(num_tokens, 2 * intermediate_size, dtype=dtype, device="cuda") + 1e-6
        )

        scale_ub = (
            torch.mean(SiluAndMul.forward_native(x)).to(dtype=torch.float32, device="cuda")
            if has_scale_ub
            else None
        )

        ref_out = torch.empty(num_tokens, intermediate_size, device="cuda", dtype=FP8_DTYPE)
        ref_scales = torch.empty((x.shape[0], 1), device="cuda", dtype=torch.float32)
        baseline(ref_out, x, ref_scales, scale_ub)

        ops_out = torch.empty(num_tokens, intermediate_size, device="cuda", dtype=FP8_DTYPE)
        ops_scales = torch.empty((x.shape[0], 1), device="cuda", dtype=torch.float32)
        silu_and_mul_dynamic_per_token_quant(ops_out, x, ops_scales, scale_ub)

        torch.testing.assert_close(ref_scales, ops_scales)
        # allow 1 ULP difference
        assert (
            ref_out.view(torch.uint8).to(torch.int16)
            - ops_out.view(torch.uint8).to(torch.int16)
        ).abs().max() <= 1


