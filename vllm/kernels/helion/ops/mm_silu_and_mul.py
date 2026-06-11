# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from itertools import product
from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
import helion.language as hl

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover
    # all input property combination. Currently, dtypes are fixed. We need
    # optimization to bucket/skip some combinations
    m_size_list = [1, 2, 4, 8, 512]
    b_shape_list = [
        # Qwen3-8B
        (8192, 6144),
        (4096, 4096),
        (4096, 24576),
        (12288, 4096),
    ]

    in_dtype: torch.dtype = torch.bfloat16
    scale_dtype: torch.dtype = torch.float32
    inputs = {}
    for M, (K, N) in product(m_size_list, b_shape_list):
        scale = 1.0 / math.sqrt(K)
        a = (scale * (0.5 + torch.rand(M, K, dtype=torch.float32, device="cuda"))).to(
            in_dtype
        )
        b = (scale * (0.5 + torch.rand(N, K, dtype=torch.float32, device="cuda"))).to(
            in_dtype
        )
        b = b.t()
        bias = 0.5 * (torch.rand(N, dtype=in_dtype, device="cuda") - 0.5)

        config_key = CaseKey(
            {
                "K": K,
                "N": N,
                "M": M,
            }
        )
        inputs[config_key] = (a, b, bias)

    return inputs


_pick_cache: dict[tuple[int, int, int], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.
    Selection strategy:
      1. Find the closest K among available configs
         (exact match preferred).
      2. Find the closest N among available configs
         (exact match preferred).
      3. Among the M values tuned for that K and N, pick
         the smallest M >= the input's M. If the input is
         larger than all available Ms, fall back to the largest.
    """

    if not config_keys:
        return None

    a, b, *_ = args
    M, K = a.shape
    N = b.shape[1]

    cache_key = (M, K, N)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    configs: dict[int, dict[int, list[int]]] = {}
    for key in config_keys:
        if key.is_default():
            continue
        configs.setdefault(key["K"], {}).setdefault(key["N"], []).append(key["M"])

    if not configs:
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
        }
    )
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    M = a.shape[0]
    N = b.shape[1]
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    return c


def baseline(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    # out = torch.mm(a.to(torch.float32), b.to(torch.float32))
    # out = scale_a * out
    # out = scale_b.T * out
    # out = out.to(out_dtype)
    # if bias is not None:
    #     out = out + bias

    # out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    # torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

    out = torch.nn.functional.linear(a, b.t(), bias)
    N = b.shape[-1] // 2
    return torch.nn.functional.silu(out[:, :N]) * out[:, N:]


# Overwrite autotune_baseline_atol and autotune_baseline_rtol
# if too many configs failed due to baseline check during autotuning
@register_kernel(
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
        autotune_baseline_atol=1.0,
        autotune_baseline_rtol=5e-1,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def mm_silu_and_mul(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, 2 * N]
    bias: torch.Tensor | None = None,  # [2 * N]
) -> torch.Tensor:
    M, K = a.shape
    two_N = b.shape[1]
    assert two_N % 2 == 0
    N = two_N // 2
    hl.specialize(K)
    hl.specialize(two_N)
    hl.specialize(N)

    assert N > 0 and K > 0 and M > 0
    assert b.shape[0] == K
    assert a.dtype == b.dtype
    assert a.stride(1) == 1
    assert b.stride(0) == 1

    if bias is not None:
        assert bias.numel() == two_N

    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    acc_dtype = torch.float32

    for tile_m, tile_n in hl.tile([M, N]):
        acc_blk = hl.zeros([tile_m, tile_n.block_size * 2], acc_dtype)
        n_idx = tile_n.begin + hl.arange(tile_n.block_size)
        tile_2n = hl.join(n_idx, n_idx + N).permute(1, 0).reshape([tile_n.block_size * 2])
        for tile_k in hl.tile(K):
            b_blk = hl.load(b, [tile_k, tile_2n], extra_mask=(tile_2n < two_N)[None, :])
            acc_blk = hl.dot(
                a[tile_m, tile_k],
                b_blk,
                acc=acc_blk,
                out_dtype=acc_dtype,
            )

        if bias is not None:
            bias_blk = hl.load(bias, [tile_2n], extra_mask=(tile_2n < two_N))
            acc_blk += bias_blk

        acc_a_blk, acc_b_blk = hl.split(
            acc_blk.reshape([tile_m, 2, tile_n]).permute(0, 2, 1)
        )
        c_blk = torch.nn.functional.silu(acc_a_blk) * acc_b_blk
        c[tile_m, tile_n] = c_blk.to(c.dtype)

    return c
