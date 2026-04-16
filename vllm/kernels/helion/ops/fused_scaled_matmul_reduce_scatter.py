# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from itertools import product
from typing import Any

import helion
import helion.language as hl
import regex as re
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from helion.runtime.dist_utils import symm_mem_sync

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
    num_tokens_list = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    b_shape_list = [
        # 1.7B
        (2048, 2048),
        (1024, 2048),
	(2048, 6144),
        (3072, 2048),

        # 8B
	(4096, 3072),
        (2048, 4096),
	(4096, 12288),
	(6144, 4096),

        # 70B
        (8192, 5120),
        (4096, 8192),
        (8192, 28672),
        (14336, 8192),
    ]
    in_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    out_dtype: torch.dtype = torch.bfloat16
    inputs = {}
    for num_tokens, (hidden_size, feature_size) in product(
        num_tokens_list, b_shape_list
    ):
        scale = 1.0 / math.sqrt(hidden_size)
        a = (
            scale
            * (
                0.5
                + torch.rand(
                    num_tokens, hidden_size, dtype=torch.float32, device="cuda"
                )
            )
        ).to(in_dtype)
        b = (
            scale
            * (
                0.5
                + torch.rand(
                    feature_size, hidden_size, dtype=torch.float32, device="cuda"
                )
            )
        ).to(in_dtype)
        b = b.t()
        scale_a = 0.5 + torch.rand((num_tokens, 1), dtype=scale_dtype, device="cuda")
        scale_b = 0.5 + torch.rand((1, feature_size), dtype=scale_dtype, device="cuda")

        group_name = dist.group.WORLD.group_name
        symm_mem_buffer = symm_mem.empty(
            num_tokens, feature_size, dtype=torch.float32, device=a.device
        )  # type: ignore[unsupported-operation]
        symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group_name)
        rank = symm_mem_hdl.rank
        world_size = symm_mem_hdl.world_size
        signal_pad_ptrs = symm_mem_hdl.signal_pad_ptrs_dev

        config_key = (
            f"hidden_size_{hidden_size}_"
            f"feature_size_{feature_size}_num_tokens_{num_tokens}"
        )
        inputs[config_key] = (
            a,
            b,
            scale_a,
            scale_b,
            out_dtype,
            symm_mem_buffer,
            signal_pad_ptrs,
            rank,
            world_size,
            group_name,
            None,
        )

    return inputs


def pick_config(args: tuple[Any, ...], config_keys: list[str]) -> str | None:
    """Pick the best pre-tuned config for the given input shape.
    Selection strategy:
      1. Find the closest hidden_size among available configs
         (exact match preferred).
      2. Find the closest feature_size among available configs
         (exact match preferred).
      3. Among the num_tokens values tuned for that hidden_size and feature_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.
    Config keys must be "default" or follow the format
    "hidden_size_{int}_feature_size_{int}_num_tokens_{int}".
    """

    if not config_keys:
        return None

    a, b, *_ = args
    num_tokens, hidden_size = a.shape
    feature_size = b.shape[1]

    configs: dict[int, dict[int, list[int]]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(
            r"hidden_size_(\d+)_feature_size_(\d+)_num_tokens_(\d+)", key
        )
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format 'hidden_size_{{int}}_"
                f"feature_size_{{int}}_num_tokens_{{int}}'"
            )
        hidden_size_str, feature_size_str, num_tokens_str = match.groups()
        configs.setdefault(int(hidden_size_str), {}).setdefault(
            int(feature_size_str), []
        ).append(int(num_tokens_str))

    if not configs:
        return "default" if "default" in config_keys else None

    best_hidden_size = min(configs, key=lambda s: abs(s - hidden_size))
    best_feature_size = min(
        configs[best_hidden_size], key=lambda s: abs(s - feature_size)
    )
    available_num_tokens = sorted(configs[best_hidden_size][best_feature_size])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens), available_num_tokens[-1]
    )

    return (
        f"hidden_size_{best_hidden_size}_feature_size_"
        f"{best_feature_size}_num_tokens_{best_num_tokens}"
    )


def fake_impl(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    symm_mem_buffer: torch.Tensor,
    signal_pad_ptrs: torch.Tensor,
    rank: hl.constexpr,
    world_size: hl.constexpr,
    group_name: hl.constexpr,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    M = a.shape[0]
    N = b.shape[1]
    M_scatter = M // world_size
    c = torch.empty((M_scatter, N), dtype=out_dtype, device=a.device)

    return c


@register_kernel(
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_atol=1.0,
        autotune_baseline_rtol=5e-1,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def fused_scaled_matmul_reduce_scatter(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    symm_mem_buffer: torch.Tensor,
    signal_pad_ptrs: torch.Tensor,
    rank: hl.constexpr,
    world_size: hl.constexpr,
    group_name: hl.constexpr,
    bias: torch.Tensor | None = None,
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

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(1, -1) if scale_b.dim() <= 1 else scale_b

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[0] == 1 and (scale_b.shape[1] == 1 or scale_b.shape[1] == N)
    assert out_dtype.is_floating_point

    if bias is not None:
        assert bias.numel() == N and bias.dtype == out_dtype

    assert M % world_size == 0
    M_scatter = M // world_size  # type: ignore[unsupported-operation]

    c = torch.empty((M_scatter, N), dtype=out_dtype, device=a.device)
    acc_dtype = torch.float32 if a.is_floating_point() else torch.int32

    buffer_tuple = torch.ops.symm_mem.get_remote_tensors(symm_mem_buffer, group_name)

    scatter_begin = rank * M_scatter  # type: ignore[unsupported-operation]
    scatter_end = scatter_begin + M_scatter  # type: ignore[unsupported-operation]

    for tile_m, tile_n in hl.tile([M, N]):
        acc = hl.zeros([tile_m, tile_n], acc_dtype)
        for tile_k in hl.tile(K):
            acc = hl.dot(
                a[tile_m, tile_k],
                b[tile_k, tile_n],
                acc=acc,
                out_dtype=acc_dtype,
            )

        acc = acc.to(torch.float32)
        scale_a_mask = (tile_m.index < scale_a.shape[0])[:, None]
        scale_a_blk = torch.where(scale_a_mask, scale_a[tile_m, :], scale_a[0, 0])
        acc = scale_a_blk * acc

        scale_b_mask = (tile_n.index < scale_b.shape[1])[None, :]
        scale_b_blk = torch.where(scale_b_mask, scale_b[:, tile_n], scale_b[0, 0])
        acc = acc * scale_b_blk

        if bias is not None:
            acc += bias[tile_n]

        symm_mem_buffer[tile_m, tile_n] = acc

        hl.triton_kernel(
            symm_mem_sync,
            args=(
                signal_pad_ptrs,
                None,
                rank,
                world_size,
                True,
                True,
            ),
            output_like=None,
        )

        if tile_m.begin < scatter_end and tile_m.end > scatter_begin:  # type: ignore[unsupported-operation]
            overlap_mask = (tile_m.index >= scatter_begin) & (
                tile_m.index < scatter_end
            )
            acc_reduce = hl.zeros(
                [tile_m, tile_n], dtype=torch.float32, device=a.device
            )
            for remote_buffer in buffer_tuple:
                remote_blk = hl.load(
                    remote_buffer, [tile_m, tile_n], extra_mask=overlap_mask[:, None]
                )
                acc_reduce += remote_blk

            hl.store(
                c,
                [tile_m.index - scatter_begin, tile_n],
                acc_reduce.to(out_dtype),
                extra_mask=overlap_mask[:, None],
            )

        hl.triton_kernel(
            symm_mem_sync,
            args=(
                signal_pad_ptrs,
                None,
                rank,
                world_size,
                True,
                False,
            ),
            output_like=None,
        )

    return c


def fused_scaled_matmul_reduce_scatter_dispatch(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    group_name: hl.constexpr,
    bias: torch.Tensor | None = None,
):
    M, K = a.shape
    N = b.shape[1]
    symm_mem_buffer = symm_mem.empty(M, N, dtype=torch.float32, device=a.device)
    symm_mem_hdl = symm_mem.rendezvous(symm_mem_buffer, group_name)
    rank = symm_mem_hdl.rank
    world_size = symm_mem_hdl.world_size
    signal_pad_ptrs = symm_mem_hdl.signal_pad_ptrs_dev

    return fused_scaled_matmul_reduce_scatter(
        a,
        b,
        scale_a,
        scale_b,
        out_dtype,
        symm_mem_buffer,
        signal_pad_ptrs,
        rank,
        world_size,
        group_name,
    )


def baseline(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    symm_mem_buffer: torch.Tensor,
    signal_pad_ptrs: torch.Tensor,
    rank: hl.constexpr,
    world_size: hl.constexpr,
    group_name: hl.constexpr,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
        a,
        b,
        scale_a,
        scale_b,
        "sum",
        0,
        0,
        group_name,
        output_shape=[a.shape[0], b.shape[1]],
        out_dtype=out_dtype,
    )
