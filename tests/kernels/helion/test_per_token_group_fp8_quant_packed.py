# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the per_token_group_fp8_quant_packed helion kernel

Run `pytest tests/kernels/helion/test_per_token_group_fp8_quant_packed.py`.
"""

from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.per_token_group_fp8_quant_packed import (
    _pick_cache,
    baseline,
    per_token_group_fp8_quant_packed,
    pick_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


def _generate_fake_input(
    num_tokens: int, hidden_size: int, group_size: int
) -> tuple[Any, ...]:
    with FakeTensorMode():
        input = torch.randn(
            (num_tokens, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        output_q = torch.empty(input.shape, device=input.device, dtype=FP8_DTYPE)
        num_groups_per_row = hidden_size // group_size
        k_num_packed = (num_groups_per_row + 3) // 4
        tma_aligned_num_tokens = ((num_tokens + 3) // 4) * 4
        output_s_packed = torch.empty_strided(
            (num_tokens, k_num_packed),
            (1, tma_aligned_num_tokens),
            device=input.device,
            dtype=torch.int32,
        )
        fp8_min, fp8_max = get_fp8_min_max()
        eps = 1e-10
        args = (
            input,
            output_q,
            output_s_packed,
            group_size,
            eps,
            fp8_min,
            fp8_max,
        )
        return args


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestPerTokenGroupFp8QuantPackedConfigPicker:
    def setup_method(self):
        _pick_cache.clear()

    def test_config_picker_exact_match(self):
        config_keys = [
            CaseKey({"hidden_size": 2048, "group_size": 64, "num_tokens": 16}),
            CaseKey({"hidden_size": 4096, "group_size": 128, "num_tokens": 16}),
        ]

        args = _generate_fake_input(16, 4096, 128)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"hidden_size": 4096, "group_size": 128, "num_tokens": 16}
        )

    def test_config_picker_closest_match(self):
        config_keys = [
            CaseKey({"hidden_size": 2048, "group_size": 64, "num_tokens": 16}),
            CaseKey({"hidden_size": 2048, "group_size": 64, "num_tokens": 32}),
            CaseKey({"hidden_size": 2048, "group_size": 128, "num_tokens": 16}),
            CaseKey({"hidden_size": 2048, "group_size": 128, "num_tokens": 32}),
            CaseKey({"hidden_size": 4096, "group_size": 64, "num_tokens": 16}),
            CaseKey({"hidden_size": 4096, "group_size": 64, "num_tokens": 32}),
            CaseKey({"hidden_size": 4096, "group_size": 128, "num_tokens": 16}),
            CaseKey({"hidden_size": 4096, "group_size": 128, "num_tokens": 32}),
        ]

        args = _generate_fake_input(20, 3000, 70)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"hidden_size": 2048, "group_size": 64, "num_tokens": 32}
        )

    def test_config_picker_no_configs(self):
        config_keys: list[dict] = []

        args = _generate_fake_input(16, 4096, 128)
        selected_key = pick_config(args, config_keys)
        assert selected_key is None

    def test_config_picker_fallback_to_largest(self):
        config_keys = [
            CaseKey({"hidden_size": 2048, "group_size": 64, "num_tokens": 16}),
            CaseKey({"hidden_size": 2048, "group_size": 64, "num_tokens": 32}),
            CaseKey({"hidden_size": 2048, "group_size": 128, "num_tokens": 16}),
            CaseKey({"hidden_size": 2048, "group_size": 128, "num_tokens": 32}),
            CaseKey({"hidden_size": 4096, "group_size": 64, "num_tokens": 16}),
            CaseKey({"hidden_size": 4096, "group_size": 64, "num_tokens": 32}),
            CaseKey({"hidden_size": 4096, "group_size": 128, "num_tokens": 16}),
            CaseKey({"hidden_size": 4096, "group_size": 128, "num_tokens": 32}),
        ]

        args = _generate_fake_input(64, 8192, 256)
        selected_key = pick_config(args, config_keys)
        assert selected_key == CaseKey(
            {"hidden_size": 4096, "group_size": 128, "num_tokens": 32}
        )


class TestPerTokenGroupFp8QuantPackedCorrectness:
    @pytest.mark.parametrize(
        "num_tokens,hidden_size,group_size",
        [
            # No padding: mn=4 (mult of 4), groups_per_row=56 (mult of 4)
            (4, 7168, 128),
            # MN padding only: mn=1, tma_aligned_mn=4
            (1, 7168, 128),
            # MN padding only: mn=3, tma_aligned_mn=4
            (3, 7168, 128),
            # K padding only: groups_per_row=5 (5%4=1)
            (4, 640, 128),
            # K padding only: groups_per_row=6 (6%4=2)
            (4, 768, 128),
            # Single packed column, no padding: k_num_packed=1, mn%4=0
            (4, 384, 128),
            # Both MN and K padding
            (1, 384, 128),
            (3, 640, 128),
            # Larger shapes with no padding
            (64, 7168, 128),
            (128, 14336, 128),
            # Larger shapes with padding
            (127, 7168, 128),
            (253, 640, 128),
        ],
    )
    @pytest.mark.parametrize("padded_output_q", [False, True])
    @pytest.mark.parametrize("poisoned_outputs", [False, True])
    @pytest.mark.skipif(
        not current_platform.is_cuda(), reason="DeepGEMM not available on this platform"
    )
    def test_per_token_group_fp8_quant_packed(
        self,
        num_tokens: int,
        hidden_size: int,
        group_size: int,
        padded_output_q: bool,
        poisoned_outputs: bool,
    ):
        skip_if_platform_unsupported("per_token_group_fp8_quant_packed")

        device = "cuda"
        torch.manual_seed(42)
        fp8_min, fp8_max = get_fp8_min_max()
        eps = 1e-10

        input = (
            torch.randn((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)
            * 8
        )

        num_groups_per_row = hidden_size // group_size
        k_num_packed = (num_groups_per_row + 3) // 4
        tma_aligned_num_tokens = ((num_tokens + 3) // 4) * 4
        num_scale_elems = num_tokens + (k_num_packed - 1) * tma_aligned_num_tokens

        if tma_aligned_num_tokens == num_tokens and padded_output_q:
            # test case already covered, skip it
            return

        if padded_output_q:
            ref_q = torch.empty((tma_aligned_num_tokens, hidden_size), device=device, dtype=FP8_DTYPE)
        else:
            ref_q = torch.empty(input.shape, device=device, dtype=FP8_DTYPE)

        ref_s = torch.empty_strided(
            (num_tokens, k_num_packed),
            (1, tma_aligned_num_tokens),
            device=device,
            dtype=torch.int32,
        )
        ops_s = torch.empty_strided(
            ref_s.shape,
            ref_s.stride(),
            device=ref_s.device,
            dtype=ref_s.dtype,
        )

        if poisoned_outputs:
            ref_q.view(torch.uint8).fill_(0xFF)
            torch.as_strided(ref_s, (num_scale_elems,), (1,)).fill_(0x7F7F7F7F)
            torch.as_strided(ops_s, (num_scale_elems,), (1,)).fill_(0x7F7F7F7F)

        ops_q = ref_q.clone()

        baseline(
            input,
            ref_q,
            ref_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
        )
        per_token_group_fp8_quant_packed(
            input,
            ops_q,
            ops_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
        )

        # Verify packed scales (valid exponents + padding zeros).
        ref_s = torch.as_strided(ref_s, (num_scale_elems,), (1,)).cpu()
        ops_s = torch.as_strided(ops_s, (num_scale_elems,), (1,)).cpu()

        assert torch.equal(ops_s, ref_s), (
            f"Packed scale storage mismatch.\n"
            f"First diff at index "
            f"{(ops_s != ref_s).nonzero(as_tuple=True)[0][0].item()}"
        )

        assert torch.equal(ops_q, ref_q), "Quantized output mismatch"


    @pytest.mark.skipif(
        not current_platform.is_cuda(), reason="DeepGEMM not available on this platform"
    )
    def test_per_token_group_quant_fp8_packed_all_zero(self):
        """All-zero input must produce well-defined UE8M0 scale bytes via the eps
        floor in the kernel's UE8M0 path. Locks down the all-zero behavior before
        optimization.

        The CUDA kernel computes:
            y_s = eps / fp8_max
            y_s = exp2(ceil(log2(fmax(y_s, 1e-10))))
        For all-zero input, eps/fp8_max < 1e-10, so the inner fmax clamps back to
        1e-10, giving exp2(ceil(log2(1e-10))) = exp2(-33) => UE8M0 byte 0x5E (94).
        """

        skip_if_platform_unsupported("per_token_group_fp8_quant_packed")

        device = "cuda"
        torch.manual_seed(42)
        fp8_min, fp8_max = get_fp8_min_max()
        eps = 1e-10

        num_tokens, hidden_size, group_size = 4, 7168, 128
        input = torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

        ops_q = torch.empty(input.shape, device=device, dtype=FP8_DTYPE)

        num_groups_per_row = hidden_size // group_size
        k_num_packed = (num_groups_per_row + 3) // 4
        tma_aligned_num_tokens = ((num_tokens + 3) // 4) * 4
        num_scale_elems = num_tokens + (k_num_packed - 1) * tma_aligned_num_tokens

        ops_s = torch.empty_strided(
            (num_tokens, k_num_packed),
            (1, tma_aligned_num_tokens),
            device=device,
            dtype=torch.int32,
        )

        per_token_group_fp8_quant_packed(
            input,
            ops_q,
            ops_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
        )

        # Quantized values must be all zero.
        assert torch.equal(
            ops_q.view(torch.uint8),
            torch.zeros_like(ops_q, dtype=torch.uint8),
        ), "All-zero input should produce all-zero FP8 output"

        expected_exp_byte = 0x5E
        ref_s = torch.zeros(num_scale_elems, dtype=torch.int32, device="cpu")
        for row in range(num_tokens):
            for g in range(num_groups_per_row):
                pack_col = g // 4
                pos = g % 4
                idx = pack_col * tma_aligned_num_tokens + row
                ref_s[idx] |= expected_exp_byte << (pos * 8)

        ops_s = torch.as_strided(ops_s, (num_scale_elems,), (1,)).cpu()
        assert torch.equal(ops_s, ref_s), "All-zero scale bytes mismatch"


    @pytest.mark.skipif(
        not current_platform.is_cuda(), reason="DeepGEMM not available on this platform"
    )
    def test_per_token_group_quant_fp8_packed_mantissa_rounds_up(self):
        """Inputs whose absmax/max_8bit produces a non-power-of-2 force the
        mantissa-rounding-up branch (exp_byte += 1). Locks down this behavior
        before optimization."""

        skip_if_platform_unsupported("per_token_group_fp8_quant_packed")

        device = "cuda"
        torch.manual_seed(42)
        fp8_min, fp8_max = get_fp8_min_max()
        eps = 1e-10

        num_tokens, hidden_size, group_size = 4, 7168, 128

        # Build a tensor whose per-group absmax = 1.5 * fp8_max * 2^k for various k.
        # fp8_max = torch.finfo(torch.float8_e4m3fn).max = 448.0.
        # Then absmax/fp8_max = 1.5 * 2^k -> non-zero mantissa, triggers ceil
        # rounding to 2^(k+1). Use k=0 for simplicity; the bf16 representation of
        # 1.5*448=672.0 is exact.
        input = torch.full(
            (num_tokens, hidden_size),
            672.0,
            device=device,
            dtype=torch.bfloat16,
        )

        ref_q = torch.empty(input.shape, device=device, dtype=FP8_DTYPE)
        ops_q = ref_q.clone()

        num_groups_per_row = hidden_size // group_size
        k_num_packed = (num_groups_per_row + 3) // 4
        tma_aligned_num_tokens = ((num_tokens + 3) // 4) * 4
        num_scale_elems = num_tokens + (k_num_packed - 1) * tma_aligned_num_tokens

        ref_s = torch.empty_strided(
            (num_tokens, k_num_packed),
            (1, tma_aligned_num_tokens),
            device=device,
            dtype=torch.int32,
        )

        ops_s = torch.empty_strided(
            ref_s.shape,
            ref_s.stride(),
            device=ref_s.device,
            dtype=ref_s.dtype,
        )
        ops_s.copy_(ref_s)

        baseline(
            input,
            ref_q,
            ref_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
        )
        per_token_group_fp8_quant_packed(
            input,
            ops_q,
            ops_s,
            group_size,
            eps,
            fp8_min,
            fp8_max,
        )

        # Verify packed scales (valid exponents + padding zeros).
        ref_s = torch.as_strided(ref_s, (num_scale_elems,), (1,)).cpu()
        ops_s = torch.as_strided(ops_s, (num_scale_elems,), (1,)).cpu()

        assert torch.equal(ops_s, ref_s), "Scale bytes mismatch"
        assert torch.equal(ops_q, ref_q), "Quantized output mismatch"

class TestPerTokenGroupFp8QuantPackedIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "per_token_group_fp8_quant_packed" in registered_kernels

        kernel_wrapper = registered_kernels["per_token_group_fp8_quant_packed"]
        assert kernel_wrapper.op_name == "per_token_group_fp8_quant_packed"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args == ["output_q", "output_s_packed"]

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("per_token_group_fp8_quant_packed")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["per_token_group_fp8_quant_packed"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_fake_input(16, 4096, 128)
        assert fake_impl(*args) is None
