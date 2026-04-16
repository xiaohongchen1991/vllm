# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused_scaled_matmul_reduce_scatter helion kernel

Run `pytest tests/kernels/helion/test_fused_scaled_matmul_reduce_scatter.py`.
"""

from typing import Any

import pytest
import torch

from tests.kernels.helion.utils import skip_if_platform_unsupported
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.fused_scaled_matmul_reduce_scatter import (
    pick_config,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


def _generate_input(
    num_tokens: int, hidden_size: int, feature_size: int
) -> tuple[Any, ...]:
    in_dtype = current_platform.fp8_dtype()
    a = torch.randn(num_tokens, hidden_size, dtype=torch.float32, device="cuda").to(
        in_dtype
    )
    b = torch.randn(feature_size, hidden_size, dtype=torch.float32, device="cuda").to(
        in_dtype
    )
    b = b.t()
    scale_a = torch.randn(num_tokens, 1, dtype=torch.float32, device="cuda")
    scale_b = torch.randn(feature_size, 1, dtype=torch.float32, device="cuda")
    out_dtype = torch.bfloat16
    symm_mem_buffer = torch.empty(
        num_tokens, feature_size, dtype=out_dtype, device="cuda"
    )
    signal_pad_ptrs = torch.empty(0)

    args = (
        a,
        b,
        scale_a,
        scale_b,
        out_dtype,
        symm_mem_buffer,
        signal_pad_ptrs,
        0,
        1,
        "group_name",
        None,
    )
    return args


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestFusedScaledMatmulReduceScatterConfigPicker:
    def test_config_picker_exact_match(self):
        config_keys = [
            "default",
            "hidden_size_2048_feature_size_4096_num_tokens_16",
            "hidden_size_4096_feature_size_6144_num_tokens_16",
        ]

        args = _generate_input(16, 4096, 6144)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_4096_feature_size_6144_num_tokens_16"

    def test_config_picker_closest_match(self):
        config_keys = [
            "default",
            "hidden_size_2048_feature_size_4096_num_tokens_16",
            "hidden_size_2048_feature_size_4096_num_tokens_32",
            "hidden_size_2048_feature_size_6144_num_tokens_16",
            "hidden_size_2048_feature_size_6144_num_tokens_32",
            "hidden_size_4096_feature_size_4096_num_tokens_16",
            "hidden_size_4096_feature_size_4096_num_tokens_32",
            "hidden_size_4096_feature_size_6144_num_tokens_16",
            "hidden_size_4096_feature_size_6144_num_tokens_32",
        ]

        args = _generate_input(20, 3000, 500)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_2048_feature_size_4096_num_tokens_32"

    def test_config_picker_fallback_to_default(self):
        config_keys = ["default"]

        args = _generate_input(16, 4096, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "default"

    def test_config_picker_no_configs(self):
        config_keys: list[str] = []

        args = _generate_input(16, 4096, 4096)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            "default",
            "hidden_size_2048_feature_size_4096_num_tokens_16",
            "hidden_size_2048_feature_size_4096_num_tokens_32",
            "hidden_size_2048_feature_size_6144_num_tokens_16",
            "hidden_size_2048_feature_size_6144_num_tokens_32",
            "hidden_size_4096_feature_size_4096_num_tokens_16",
            "hidden_size_4096_feature_size_4096_num_tokens_32",
            "hidden_size_4096_feature_size_6144_num_tokens_16",
            "hidden_size_4096_feature_size_6144_num_tokens_32",
        ]

        args = _generate_input(64, 8192, 7000)
        selected_key = pick_config(args, config_keys)
        assert selected_key == "hidden_size_4096_feature_size_6144_num_tokens_32"

    def test_config_picker_malformed_key_raises(self):
        config_keys = [
            "bad_key",
        ]

        args = _generate_input(16, 4096, 4096)
        with pytest.raises(ValueError):
            pick_config(args, config_keys)


class TestFusedScaledMatmulReduceScatterIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "fused_scaled_matmul_reduce_scatter" in registered_kernels

        kernel_wrapper = registered_kernels["fused_scaled_matmul_reduce_scatter"]
        assert kernel_wrapper.op_name == "fused_scaled_matmul_reduce_scatter"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args is None

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("fused_scaled_matmul_reduce_scatter")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["fused_scaled_matmul_reduce_scatter"]
        fake_impl = kernel_wrapper._fake_impl

        a, b, scale_a, scale_b, out_dtype, *rest = _generate_input(16, 4096, 4096)
        fake_output = fake_impl(a, b, scale_a, scale_b, out_dtype, *rest)

        assert fake_output.shape[0] == a.shape[0]
        assert fake_output.shape[1] == b.shape[1]
        assert fake_output.dtype == out_dtype
        assert fake_output.device == a.device
