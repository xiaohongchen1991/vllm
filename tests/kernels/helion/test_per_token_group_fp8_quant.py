# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the rms_norm_dynamic_per_token_quant helion kernel

Run `pytest tests/kernels/helion/test_per_token_group_fp8_quant.py`.
"""

from unittest.mock import patch

import pytest
import torch

from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.per_token_group_fp8_quant import (
    per_token_group_fp8_quant,
)
from vllm.model_executor.layers.quantization.utils import fp8_utils
from vllm.utils.import_utils import has_helion

config_manager = ConfigManager()

if not has_helion():
    pytest.skip(
        "Helion is not installed. Install with: pip install vllm[helion]",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "shape", [(31, 128), (32, 128), (63, 256), (64, 256), (16, 512), (2048, 5120)]
)
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("tma_aligned", [False, True])
@pytest.mark.parametrize("scale_ue8m0", [False, True])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_per_token_group_fp8_quant(
    shape, column_major: bool, tma_aligned: bool, scale_ue8m0: bool, group_size: int
):
    device = "cuda"

    torch.manual_seed(42)
    num_tokens, hidden_dim = shape

    x = torch.randn((num_tokens, hidden_dim), device=device, dtype=torch.bfloat16) * 8

    # cuda path
    out_q, scale = fp8_utils.per_token_group_quant_fp8(
        x,
        group_size,
        column_major_scales=column_major,
        tma_aligned_scales=tma_aligned,
        use_ue8m0=scale_ue8m0,
    )

    with patch.object(
        torch.ops._C, "per_token_group_fp8_quant", new=per_token_group_fp8_quant
    ):
        ref_q, ref_s = fp8_utils.per_token_group_quant_fp8(
            x,
            group_size,
            column_major_scales=column_major,
            tma_aligned_scales=tma_aligned,
            use_ue8m0=scale_ue8m0,
        )

    assert torch.allclose(scale, ref_s)
    # allow 1 ULP difference
    assert (
        ref_q.view(torch.uint8).to(torch.int16)
        - out_q.view(torch.uint8).to(torch.int16)
    ).abs().max() <= 1
