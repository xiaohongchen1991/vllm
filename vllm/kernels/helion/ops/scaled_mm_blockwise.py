# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import product
from typing import Any

import helion
import helion.language as hl
import regex as re
import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def generate_inputs() -> dict[str, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover
    # all input property combination. Currently, dtypes are fixed. We need
    # optimization to bucket/skip some combinations
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    hidden_size_list = [2048, 4096, 8192]
    in_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    out_dtype: torch.dtype = torch.bfloat16
    group_m = 1
    group_k = 128
    group_n = 128
    # generate_inputs is used for benchmarking purpose as well.
    # Currently, bias not yet supported by blockwise cutlass_scaled_mm.
    # Set it to None to avoid failure during benchmarking.
    bias = None
    inputs = {}
    for num_tokens, hidden_size in product(num_tokens_list, hidden_size_list):
        feature_size = hidden_size * 4
        a = (
            0.25
            * torch.rand(num_tokens, hidden_size, dtype=torch.float32, device="cuda")
        ).to(in_dtype)
        b = (
            0.25
            * torch.rand(feature_size, hidden_size, dtype=torch.float32, device="cuda")
        ).to(in_dtype)
        b = b.t()
        num_group_m = num_tokens // group_m
        num_group_k = hidden_size // group_k
        num_group_n = feature_size // group_n
        scale_a = 0.25 * torch.rand(
            num_group_m, num_group_k, dtype=scale_dtype, device="cuda"
        )
        scale_b = 0.25 * torch.rand(
            num_group_k, num_group_n, dtype=scale_dtype, device="cuda"
        )
        scale_a = scale_a.t().contiguous().t()
        scale_b = scale_b.t().contiguous().t()

        config_key = f"hidden_size_{hidden_size}_num_tokens_{num_tokens}"
        inputs[config_key] = (
            a,
            b,
            scale_a,
            scale_b,
            group_m,
            group_k,
            group_n,
            out_dtype,
            bias,
        )

    return inputs


def pick_config(args: tuple[Any, ...], config_keys: list[str]) -> str | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest hidden_size among available configs
         (exact match preferred).
      2. Among the num_tokens values tuned for that hidden_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.

    Config keys must be "default" or follow the format
    "hidden_size_{int}_num_tokens_{int}".
    """

    if not config_keys:
        return None

    a, *_ = args
    num_tokens, hidden_size = a.shape

    configs: dict[int, list[int]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(r"hidden_size_(\d+)_num_tokens_(\d+)", key)
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format 'hidden_size_{{int}}_num_tokens_{{int}}'"
            )
        hidden_size_str, num_tokens_str = match.groups()
        configs.setdefault(int(hidden_size_str), []).append(int(num_tokens_str))

    if not configs:
        return "default" if "default" in config_keys else None

    best_hidden_size = min(configs, key=lambda s: abs(s - hidden_size))
    available_num_tokens = sorted(configs[best_hidden_size])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens), available_num_tokens[-1]
    )

    return f"hidden_size_{best_hidden_size}_num_tokens_{best_num_tokens}"


def baseline(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [num_group_m, num_group_k]
    scale_b: torch.Tensor,  # [num_group_k, num_group_n]
    group_m: int,
    group_k: int,
    group_n: int,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    def group_broadcast(t, shape):
        for i, s in enumerate(shape):
            if t.shape[i] != s and t.shape[i] != 1:
                assert s % t.shape[i] == 0
                t = (
                    t.unsqueeze(i + 1)
                    .expand(*t.shape[: i + 1], s // t.shape[i], *t.shape[i + 1 :])
                    .flatten(i, i + 1)
                )
        return t

    scale_a = group_broadcast(scale_a, a.shape)
    scale_b = group_broadcast(scale_b, b.shape)

    out = torch.mm(
        (scale_a * a.to(dtype=torch.float32)), (scale_b * b.to(dtype=torch.float32))
    ).to(out_dtype)

    if bias is not None:
        out = out + bias

    return out


def fake_impl(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [num_group_m, num_group_k]
    scale_b: torch.Tensor,  # [num_group_k, num_group_n]
    group_m: int,
    group_k: int,
    group_n: int,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    M = a.shape[0]
    N = b.shape[1]
    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    return c


@register_kernel(
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_atol=1.0,
        autotune_baseline_rtol=1e-1,
        autotune_baseline_fn=baseline,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def scaled_mm_blockwise(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [num_group_m, num_group_k]
    scale_b: torch.Tensor,  # [num_group_k, num_group_n]
    group_m: int,
    group_k: int,
    group_n: int,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    assert a.ndim == 2 and b.ndim == 2
    M, K = a.shape
    N = b.shape[1]
    hl.specialize(K)
    hl.specialize(N)

    assert N > 0 and K > 0 and M > 0
    assert b.shape[0] == K
    assert a.dtype == b.dtype

    assert a.stride(1) == 1
    assert b.stride(0) == 1

    if bias is not None:
        assert (
            bias.numel() == N
            and bias.dtype == out_dtype
            and bias.ndim == 1
            and bias.is_contiguous()
        )

    assert scale_a.ndim == 2 and scale_b.ndim == 2
    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()

    num_group_m, num_group_k = scale_a.shape
    num_group_n = scale_b.shape[1]
    hl.specialize(num_group_k)
    hl.specialize(num_group_n)

    assert M % num_group_m == 0
    assert K % num_group_k == 0
    assert N % num_group_n == 0

    # scale_a group shape must be [1, 128]
    # scale_b group shape must be [128, 128]
    assert scale_b.shape[0] == num_group_k
    assert M // num_group_m == group_m
    assert K // num_group_k == group_k
    assert N // num_group_n == group_n

    hl.specialize(group_m)
    hl.specialize(group_k)
    hl.specialize(group_n)

    assert out_dtype.is_floating_point

    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    accumulator_dtype = torch.float32

    for tile_gm, tile_gn in hl.tile([num_group_m, num_group_n]):
        gm_idx = tile_gm.index
        gn_idx = tile_gn.index

        m_offset = hl.arange(group_m)
        n_offset = hl.arange(group_n)

        m_idx = gm_idx[:, None] * group_m + m_offset[None, :]
        n_idx = gn_idx[:, None] * group_n + n_offset[None, :]

        m_idx_1d = tile_gm.begin * group_m + hl.arange(tile_gm.block_size * group_m)
        n_idx_1d = tile_gn.begin * group_n + hl.arange(tile_gn.block_size * group_n)

        # shape: [tile_gm * group_m, tile_gn * group_n]
        accumulator = hl.zeros(
            [tile_gm.block_size * group_m, tile_gn.block_size * group_n],
            accumulator_dtype,
        )
        for tile_gk in hl.tile(num_group_k):
            gk_idx = tile_gk.index
            k_offset = hl.arange(group_k)
            k_idx = gk_idx[:, None] * group_k + k_offset[None, :]

            a_mask = (gm_idx < num_group_m)[:, None, None, None] & (
                gk_idx < num_group_k
            )[None, None, :, None]
            b_mask = (gk_idx < num_group_k)[:, None, None, None] & (
                gn_idx < num_group_n
            )[None, None, :, None]
            # shape: [tile_gm, group_m, tile_gk, group_k]
            a_blk = hl.load(
                a, [m_idx[:, :, None, None], k_idx[None, None, :, :]], extra_mask=a_mask
            )
            # shape: [tile_gk, group_k, tile_gn, group_n]
            b_blk = hl.load(
                b, [k_idx[:, :, None, None], n_idx[None, None, :, :]], extra_mask=b_mask
            )

            scale_a_blk = scale_a[tile_gm, tile_gk]
            scale_b_blk = scale_b[tile_gk, tile_gn]

            a_blk = scale_a_blk[:, None, :, None] * a_blk.to(torch.float32)
            b_blk = scale_b_blk[:, None, :, None] * b_blk.to(torch.float32)

            # flatten to 2d
            a_blk = a_blk.flatten(2, 3).flatten(0, 1)
            b_blk = b_blk.flatten(0, 1).flatten(1, 2)

            accumulator = hl.dot(
                a_blk,
                b_blk,
                acc=accumulator,
                out_dtype=accumulator_dtype,
            )

        # shape: [tile_gm * group_m, tile_gn * group_n]
        c_blk = accumulator.to(out_dtype)

        if bias is not None:
            bias_blk = hl.load(bias, [n_idx_1d], extra_mask=(n_idx_1d < N))
            c_blk += bias_blk

        c_mask = (m_idx_1d < M)[:, None] & (n_idx_1d < N)[None, :]
        hl.store(c, [m_idx_1d[:, None], n_idx_1d[None, :]], c_blk, extra_mask=c_mask)

    return c
