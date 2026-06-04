# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rms_norm_per_block_quant_packed helion kernel

Run `pytest tests/kernels/helion/test_rms_norm_per_block_quant_packed.py`.
"""

import itertools
from typing import Any

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from tests.kernels.helion.utils import skip_if_platform_unsupported
from tests.kernels.quant_utils import FP8_DTYPE
from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.rms_norm_per_block_quant_packed import (
    _pick_cache,
    baseline,
    pick_config,
    rms_norm_per_block_quant_packed,
)
from vllm.utils.import_utils import has_helion
from vllm.utils.torch_utils import set_random_seed

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
        result = torch.empty(input.shape, device=input.device, dtype=FP8_DTYPE)
        groups_per_row = hidden_size // group_size
        packed_groups_per_row = (groups_per_row + 3) // 4
        tma_aligned_num_tokens = ((num_tokens + 3) // 4) * 4
        scale = torch.empty_strided(
            (num_tokens, packed_groups_per_row),
            (1, tma_aligned_num_tokens),
            device=input.device,
            dtype=torch.int32,
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
        args = (
            result,
            input,
            weight,
            scale,
            epsilon,
            residual,
            group_size,
        )
        return args


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


class TestRmsNormPerBlockQuantPackedConfigPicker:
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


DTYPES = [torch.bfloat16, torch.float16]
QUANT_DTYPES = [FP8_DTYPE]
# Avoid combinatorial explosion with full Cartesian product
NUM_TOKENS_HIDDEN_SIZES = [
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

ADD_RESIDUAL = [False, True]
SEEDS = [0]
EPS = 1e-6


class TestRmsNormPerBlockQuantPackedCorrectness:
    @pytest.mark.parametrize("num_tokens, hidden_size", NUM_TOKENS_HIDDEN_SIZES)
    @pytest.mark.parametrize("add_residual", ADD_RESIDUAL)
    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("quant_dtype", QUANT_DTYPES)
    @pytest.mark.parametrize("seed", SEEDS)
    def test_rms_norm_per_block_quant_packed(
        self,
        num_tokens: int,
        hidden_size: int,
        add_residual: bool,
        dtype: torch.dtype,
        quant_dtype: torch.dtype,
        seed: int,
    ) -> None:
        skip_if_platform_unsupported("rms_norm_per_block_quant_packed")

        set_random_seed(seed)

        group_size = 128
        if hidden_size % group_size != 0:
            # skip
            return

        scale = 1 / (hidden_size)
        input = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda") * scale
        weight = torch.normal(
            mean=1.0, std=1.0, size=(hidden_size,), dtype=dtype, device=input.device
        )
        residual = torch.randn_like(input) * scale if add_residual else None
        groups_per_row = hidden_size // group_size

        ref_residual = residual.clone() if residual is not None else None
        ops_residual = residual.clone() if residual is not None else None
        ref_out = torch.empty(input.shape, device=input.device, dtype=quant_dtype)
        ops_out = ref_out.clone()

        packed_groups_per_row = (groups_per_row + 3) // 4
        tma_aligned_num_tokens = ((num_tokens + 3) // 4) * 4
        num_scale_elems = num_tokens + (packed_groups_per_row - 1) * tma_aligned_num_tokens

        ref_s = torch.empty_strided(
            (num_tokens, packed_groups_per_row),
            (1, tma_aligned_num_tokens),
            device=input.device,
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

        baseline(
            ref_out,
            input,
            weight,
            ref_s,
            EPS,
            ref_residual,
            group_size,
        )

        rms_norm_per_block_quant_packed(
            ops_out,
            input,
            weight,
            ops_s,
            EPS,
            ops_residual,
            group_size,
        )

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

        if add_residual:
            torch.testing.assert_close(ref_residual, ops_residual)


class TestRmsNormPerBlockQuantPackedIntegration:
    def test_kernel_registration_integration(self):
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        assert "rms_norm_per_block_quant_packed" in registered_kernels

        kernel_wrapper = registered_kernels["rms_norm_per_block_quant_packed"]
        assert kernel_wrapper.op_name == "rms_norm_per_block_quant"
        assert kernel_wrapper._config_picker is not None
        assert kernel_wrapper._mutates_args == ["result", "scale", "residual"]

    def test_fake_impl_functionality(self):
        skip_if_platform_unsupported("rms_norm_per_block_quant")
        from vllm.kernels.helion.register import get_registered_kernels

        registered_kernels = get_registered_kernels()
        kernel_wrapper = registered_kernels["rms_norm_per_block_quant_packed"]
        fake_impl = kernel_wrapper._fake_impl

        args = _generate_fake_input(16, 4096, 128)
        assert fake_impl(*args) is None
