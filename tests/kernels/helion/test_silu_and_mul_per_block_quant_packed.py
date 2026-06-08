# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the silu_and_mul_per_block_quant_packed helion kernel
Run `pytest tests/kernels/helion/test_silu_and_mul_per_block_quant_packed.py`.
"""

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from vllm.kernels.helion.case_key import CaseKey

from tests.kernels.helion.utils import skip_if_platform_unsupported
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.silu_and_mul_per_block_quant_packed import (
    _pick_cache,
    baseline,
    pick_config,
    silu_and_mul_per_block_quant_packed,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


def _generate_fake_input(
    num_tokens: int, intermediate_size: int, group_size: int
) -> tuple[Any, ...]:
    with FakeTensorMode():
        in_dtype: torch.dtype = torch.bfloat16
        out_dtype: torch.dtype = current_platform.fp8_dtype()
        scale_dtype: torch.dtype = torch.int32
        input = torch.randn(
            num_tokens, 2 * intermediate_size, device="cuda", dtype=in_dtype
        )
        result = torch.empty(
            num_tokens, intermediate_size, device=input.device, dtype=out_dtype
        )
        groups_per_row = intermediate_size // group_size
        packed_groups_per_row = (groups_per_row + 3) // 4
        tma_aligned_num_tokens = ((num_tokens + 3) // 4) * 4
        scale = torch.empty_strided(
            (num_tokens, packed_groups_per_row),
            (1, tma_aligned_num_tokens),
            device=input.device,
            dtype=scale_dtype,
        )
        args = (
            result,
            input,
            scale,
            group_size,
        )
        return args


class TestSiluAndMulPerBlockQuantPackedConfigPicker:
    def setup_method(self):
        _pick_cache.clear()

    def test_config_picker_exact_match(self):
        config_keys = [
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 16}),
        ]

        args = _generate_fake_input(16, 4096, 128)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"intermediate_size": 4096, "group_size": 128, "num_tokens": 16}
        )

    def test_config_picker_closest_match(self):
        config_keys = [
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 32}),
            CaseKey({"intermediate_size": 2048, "group_size": 128, "num_tokens": 16}),
            CaseKey({"intermediate_size": 2048, "group_size": 128, "num_tokens": 32}),
            CaseKey({"intermediate_size": 4096, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 64, "num_tokens": 32}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 32}),
        ]

        args = _generate_fake_input(20, 3000, 70)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"intermediate_size": 2048, "group_size": 64, "num_tokens": 32}
        )

    def test_config_picker_no_configs(self):
        config_keys: list[dict] = []

        args = _generate_fake_input(16, 4096, 128)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 2048, "group_size": 64, "num_tokens": 32}),
            CaseKey({"intermediate_size": 2048, "group_size": 128, "num_tokens": 16}),
            CaseKey({"intermediate_size": 2048, "group_size": 128, "num_tokens": 32}),
            CaseKey({"intermediate_size": 4096, "group_size": 64, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 64, "num_tokens": 32}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 16}),
            CaseKey({"intermediate_size": 4096, "group_size": 128, "num_tokens": 32}),
        ]

        args = _generate_fake_input(64, 8192, 256)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"intermediate_size": 4096, "group_size": 128, "num_tokens": 32}
        )


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestSiluAndMulPerBlockQuantPackedCorrectness:
    @pytest.mark.parametrize(
        "num_tokens, hidden_size",
        [
            # No padding: mn=4 (mult of 4), groups_per_row=56 (mult of 4)
            (4, 7168),
            # MN padding only: mn=1, tma_aligned_mn=4
            (1, 7168),
            # MN padding only: mn=3, tma_aligned_mn=4
            (3, 7168),
            # K padding only: groups_per_row=5 (5%4=1)
            (4, 640),
            # K padding only: groups_per_row=6 (6%4=2)
            (4, 768),
            # Single packed column, no padding: k_num_packed=1, mn%4=0
            (4, 384),
            # Both MN and K padding
            (1, 384),
            (3, 640),
            # Larger shapes with no padding
            (64, 7168),
            (128, 14336),
            # Larger shapes with padding
            (127, 7168),
            (253, 640),
        ]
    )
    @pytest.mark.parametrize("group_size", [128])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    @pytest.mark.parametrize("quant_dtype", [current_platform.fp8_dtype()])
    @pytest.mark.parametrize("seed", [0])
    def test_silu_and_mul_per_block_quant_packed(
        self,
        num_tokens: int,
        hidden_size: int,
        group_size: int,
        dtype: torch.dtype,
        quant_dtype: torch.dtype,
        seed: int,
    ) -> None:
        skip_if_platform_unsupported("silu_and_mul_per_block_quant_packed")
        set_random_seed(seed)

        if hidden_size % group_size != 0:
            return

        x = torch.randn(num_tokens, 2 * hidden_size, dtype=dtype, device="cuda")

        ref_out = torch.empty(num_tokens, hidden_size, device="cuda", dtype=quant_dtype)
        ops_out = ref_out.clone()

        groups_per_row = hidden_size // group_size
        packed_groups_per_row = (groups_per_row + 3) // 4
        tma_aligned_num_tokens = ((num_tokens + 3) // 4) * 4
        num_scale_elems = num_tokens + (packed_groups_per_row - 1) * tma_aligned_num_tokens

        ref_s = torch.empty_strided(
            (num_tokens, packed_groups_per_row),
            (1, tma_aligned_num_tokens),
            device=x.device,
            dtype=torch.int32,
        )
        ops_s = torch.empty_strided(
            ref_s.shape,
            ref_s.stride(),
            device=ref_s.device,
            dtype=ref_s.dtype,
        )

        torch.as_strided(ref_s, (num_scale_elems,), (1,)).fill_(0x7F7F7F7F)
        torch.as_strided(ops_s, (num_scale_elems,), (1,)).fill_(0x7F7F7F7F)

        baseline(ref_out, x, ref_s, group_size)

        silu_and_mul_per_block_quant_packed(ops_out, x, ops_s, group_size)

        # Verify packed scales (valid exponents + padding zeros).
        ref_s = torch.as_strided(ref_s, (num_scale_elems,), (1,)).cpu()
        ops_s = torch.as_strided(ops_s, (num_scale_elems,), (1,)).cpu()

        if not torch.equal(ops_s, ref_s):
            diff_idx = (ops_s != ref_s).nonzero(as_tuple=True)[0][0].item()

            row = diff_idx % tma_aligned_num_tokens
            packed_col = diff_idx // tma_aligned_num_tokens

            assert False, (
                f"Packed scale storage mismatch.\n"
                f"First diff at flat index {diff_idx}\n"
                f"row={row}, packed_col={packed_col}\n"
                f"ops_s = {ops_s[diff_idx].item()} "
                f"(0x{ops_s[diff_idx].item() & 0xffffffff:08x})\n"
                f"ref_s = {ref_s[diff_idx].item()} "
                f"(0x{ref_s[diff_idx].item() & 0xffffffff:08x})"
            )

        # allow 1 ULP difference
        assert (
            ref_out.view(torch.uint8).to(torch.int16)
            - ops_out.view(torch.uint8).to(torch.int16)
        ).abs().max() <= 1


class TestSiluAndMulPerBlockQuantPackedIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "silu_and_mul_per_block_quant_packed" in registered_kernels

        kernel_wrapper = registered_kernels["silu_and_mul_per_block_quant_packed"]
        assert kernel_wrapper.op_name == "silu_and_mul_per_block_quant_packed"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args == ["out", "scales"]

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("silu_and_mul_per_block_quant_packed")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["silu_and_mul_per_block_quant_packed"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_fake_input(16, 4096, 128)
        assert fake_impl(*args) is None
