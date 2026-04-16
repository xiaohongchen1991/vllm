# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the fused_scaled_matmul_reduce_scatter helion kernel

Run `torchrun --nproc-per-node=2 -m pytest \
    tests/kernels/helion/distributed/test_fused_scaled_matmul_reduce_scatter_distributed.py`.
"""

from typing import Any
from unittest import skipUnless

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_cuda import (
    SM89OrLater,
)
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    requires_cuda,
    requires_cuda_p2p_access,
    run_tests,
)

from tests.kernels.helion.utils import skip_if_platform_unsupported
from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.fused_scaled_matmul_reduce_scatter import (
    fused_scaled_matmul_reduce_scatter_dispatch,
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

    args = (a, b, scale_a, scale_b, out_dtype, None)
    return args


@pytest.fixture(autouse=True)
def reset_config_manager_singleton():
    ConfigManager.reset_instance()
    ConfigManager()
    yield
    ConfigManager.reset_instance()


MNK_FACTORS = [
    (2, 256, 128),
    (32, 256, 496),
    (64, 976, 1024),
    (64, 20496, 128),
    (512, 256, 496),
    (512, 20496, 1024),
]


@instantiate_parametrized_tests
@requires_cuda_p2p_access()
class TestFusedScaledMatmulReduceScatterCorrectness(MultiProcContinuousTest):
    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def _init_process(self):
        torch.accelerator.set_device_index(self.device)
        torch.manual_seed(42 + self.rank)
        torch.use_deterministic_algorithms(True)
        torch.set_deterministic_debug_mode("warn")
        torch.utils.deterministic.fill_uninitialized_memory = True

    @requires_cuda
    @skip_if_lt_x_gpu(2)
    @skipUnless(SM89OrLater, "Requires compute capability >= 8.9")
    @parametrize("M, N, K", MNK_FACTORS)
    @parametrize("rowwise", [True, False])
    def test_fused_scaled_matmul_reduce_scatter(
        self, M: int, N: int, K: int, rowwise: bool
    ) -> None:
        skip_if_platform_unsupported("fused_scaled_matmul_reduce_scatter")
        self._init_process()

        group = dist.group.WORLD
        rank = self.rank
        world_size = dist.get_world_size(group)

        if M % world_size != 0:
            return

        torch.manual_seed(42 + rank)
        in_dtype: torch.dtype = current_platform.fp8_dtype()
        out_dtype: torch.dtype = torch.bfloat16

        A = torch.rand(M, K, device="cuda").to(in_dtype)
        B = torch.rand(N, K, device="cuda").to(in_dtype).T

        if rowwise:
            A_scale = torch.full((M, 1), 0.1, device="cuda")
            B_scale = torch.full((1, N), 0.1, device="cuda")
        else:
            A_scale = torch.tensor(0.1, device="cuda")
            B_scale = torch.tensor(0.1, device="cuda")

        output_shape = [M, N]

        ref = torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
            A,
            B,
            A_scale,
            B_scale,
            "sum",
            0,
            0,
            group.group_name,
            output_shape,
            out_dtype=out_dtype,
        )

        act = fused_scaled_matmul_reduce_scatter_dispatch(
            A, B, A_scale, B_scale, out_dtype, group.group_name
        )

        torch.testing.assert_close(ref, act)


if __name__ == "__main__":
    run_tests()
