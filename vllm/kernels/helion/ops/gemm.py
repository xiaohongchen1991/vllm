# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from itertools import product
from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
import helion.language as hl
from helion.autotuner import PowerOfTwoFragment

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    # The Helion linear kernel is autotuned per shape.
    # m_size_list follows cudagraph_capture_sizes pattern:
    # [1, 2, 4] + range(8, 256, 8) + range(256, max_graph_size + 1, 16),
    # but is capped here to cover only small M values.
    m_size_list = [1, 2, 4, 8, 16]

    bf16: torch.dtype = torch.bfloat16

    # Each entry maps a (K, N) weight shape to a single input dtype.
    b_shape_dtype_list: list[tuple[tuple[int, int], torch.dtype]] = [
        # DSV4-Flash
        ((4096, 256), bf16),
    ]

    out_dtype: torch.dtype = torch.float32
    inputs = {}
    for M, ((K, N), in_dtype) in product(m_size_list, b_shape_dtype_list):
        scale = 1.0 / math.sqrt(K)
        a = (scale * (0.5 + torch.rand(M, K, dtype=torch.float32, device="cuda"))).to(
            in_dtype
        )
        b = (scale * (0.5 + torch.rand(N, K, dtype=torch.float32, device="cuda"))).to(
            in_dtype
        )
        b = b.t()

        config_key = CaseKey({"K": K, "N": N, "M": M, "in_dtype": str(in_dtype)})
        inputs[config_key] = (a, b, out_dtype)

    return inputs


_pick_cache: dict[tuple[int, int, int, str], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    Configs are matched within the runtime input dtype. K/N are picked by
    closest match. M is bucketed to the smallest tuned M >= runtime M.
    """

    if not config_keys:
        return None

    a, b, *_ = args

    M, K = a.shape
    N = b.shape[1]
    in_dtype = str(a.dtype)

    cache_key = (M, K, N, in_dtype)
    if cache_key in _pick_cache:
        return _pick_cache[cache_key]

    configs: dict[int, dict[int, list[int]]] = {}
    for key in config_keys:
        if key.is_default():
            continue

        if all(k in key for k in ("K", "N", "M", "in_dtype")):
            if "in_dtype" not in key or key["in_dtype"] != in_dtype:
                continue
            configs.setdefault(key["K"], {}).setdefault(key["N"], []).append(key["M"])

    if not configs:
        _pick_cache[cache_key] = None
        return None

    best_K = min(configs, key=lambda s: abs(s - K))
    best_N = min(configs[best_K], key=lambda s: abs(s - N))
    available_M = sorted(configs[best_K][best_N])
    best_M = next((m for m in available_M if m >= M), available_M[-1])

    result = CaseKey(
        {
            "K": best_K,
            "N": best_N,
            "M": best_M,
            "in_dtype": in_dtype,
        }
    )
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    out_dtype: torch.dtype,
) -> torch.Tensor:
    M = a.shape[0]
    N = b.shape[1]
    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    return c


# from vllm.model_executor.kernels.linear.cute_dsl.ll_bf16 import (
#     ll_bf16_gemm,
#     ll_bf16_gemm_kernel,
# )

# shapes = ((4096, 256),)
# _LL_BF16_WARMUP_M_RANGE = range(1, 17)
# ll_bf16_gemm_kernel.warmup(
#     shapes=shapes,
#     m_values=_LL_BF16_WARMUP_M_RANGE,
# )


def baseline(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    out_dtype: torch.dtype,
) -> torch.Tensor:
    c = torch.mm(a.to(torch.float32), b.to(torch.float32))
    c = c.to(out_dtype)
    return c
    # return ll_bf16_gemm(a, b.T, out_dtype)


# TODO(xiaohongchen1991):
# 1. Remove ProcessGroupNameNotFound from ignore_warning after fix for
# https://github.com/pytorch/helion/issues/3024 is available in vLLM
# 2. Conditionally use SwapAB trick when the fix for
# https://github.com/pytorch/helion/issues/3044 is available in vLLM


# Quantized GEMM kernels can have relatively large numerical differences
# from the baseline.
# Override autotune_baseline_atol and autotune_baseline_rtol to prevent
# excessive config failures from baseline accuracy checks during autotuning.
@register_kernel(
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
        autotune_baseline_atol=1e-1,
        autotune_baseline_rtol=1e-1,
        ignore_warnings=[
            helion.exc.TensorOperationInWrapper,
            helion.exc.ProcessGroupNameNotFound,
        ],
    ),
)  # type: ignore[misc]
def gemm(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    out_dtype: torch.dtype,
) -> torch.Tensor:
    M, K = a.shape
    N = b.shape[1]
    hl.specialize(K)
    hl.specialize(N)

    assert N > 0 and K > 0 and M > 0
    assert b.shape[0] == K
    assert a.dtype == b.dtype
    assert a.stride(1) == 1
    assert b.stride(0) == 1

    assert out_dtype.is_floating_point
    acc_dtype = torch.float32

    split_k = hl.register_tunable("split_k", PowerOfTwoFragment(1, 256))
    k_block_size = helion.next_power_of_2(helion.cdiv(K, split_k))

    if split_k > 1:
        out = torch.zeros((M, N), dtype=out_dtype, device=a.device)
    else:
        out = torch.empty((M, N), dtype=out_dtype, device=a.device)

    for tile_m, tile_n, outer_k in hl.tile(
        [M, N, K], block_size=[None, None, k_block_size]
    ):
        acc = hl.zeros([tile_n, tile_m], acc_dtype)
        for tile_k in hl.tile(outer_k.begin, outer_k.end):
            a_blk = hl.load(a, [tile_m.index[None, :], tile_k.index[:, None]])
            b_blk = hl.load(b, [tile_k.index[None, :], tile_n.index[:, None]])
            acc = hl.dot(
                b_blk,
                a_blk,
                acc=acc,
                out_dtype=acc_dtype,
            )

        acc = acc.t().to(torch.float32)
        out_blk = acc.to(out_dtype)

        if split_k == 1:
            out[tile_m, tile_n] = out_blk
        else:
            hl.atomic_add(out, [tile_m, tile_n], out_blk)

    return out
