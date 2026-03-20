# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import torch
import triton
import triton.language as tl
from helion.runtime import default_launcher as _default_launcher

import vllm.kernels.helion.ops.scaled_mm as _source_module
_BLOCK_SIZE_1 = tl.constexpr(128)
_BLOCK_SIZE_0 = tl.constexpr(64)
_BLOCK_SIZE_2 = tl.constexpr(256)

@triton.jit
def _helion_scaled_mm(a, b, c, a_stride_0, a_stride_1, b_stride_0, b_stride_1, c_stride_0, c_stride_1, N, M, K):
    # src[scaled_mm.py:150]: for tile_m, tile_n in hl.tile([M, N]):
    num_blocks_0 = tl.cdiv(N, _BLOCK_SIZE_1)
    pid_0 = tl.program_id(0) % num_blocks_0
    pid_1 = tl.program_id(0) // num_blocks_0
    offset_1 = pid_0 * _BLOCK_SIZE_1
    indices_1 = (offset_1 + tl.arange(0, _BLOCK_SIZE_1)).to(tl.int32)
    mask_1 = indices_1 < N
    offset_0 = pid_1 * _BLOCK_SIZE_0
    indices_0 = (offset_0 + tl.arange(0, _BLOCK_SIZE_0)).to(tl.int32)
    mask_0 = indices_0 < M
    # src[scaled_mm.py:151]: accumulator = hl.zeros([tile_m, tile_n], accumulator_dtype)
    accumulator = tl.zeros([_BLOCK_SIZE_0, _BLOCK_SIZE_1], tl.float32)
    # src[scaled_mm.py:152]: for tile_k in hl.tile(K):
    # src[scaled_mm.py:153]:     a_blk = a[tile_m, tile_k]
    # src[scaled_mm.py:154]:     b_blk = b[tile_k, tile_n]
    # src[scaled_mm.py:152-159]: ...
    for offset_2 in tl.range(0, tl.cast(K, tl.int32), _BLOCK_SIZE_2):
        indices_2 = offset_2 + tl.arange(0, _BLOCK_SIZE_2).to(tl.int32)
        mask_2 = indices_2 < K
        # src[scaled_mm.py:153]: a_blk = a[tile_m, tile_k]
        a_blk = tl.load(a + (indices_0[:, None] * a_stride_0 + indices_2[None, :] * a_stride_1), mask_0[:, None] & mask_2[None, :])
        # src[scaled_mm.py:154]: b_blk = b[tile_k, tile_n]
        b_blk = tl.load(b + (indices_2[:, None] * b_stride_0 + indices_1[None, :] * b_stride_1), mask_2[:, None] & mask_1[None, :])
        # src[scaled_mm.py:155]: accumulator = hl.dot(
        # src[scaled_mm.py:156]:     a_blk,
        # src[scaled_mm.py:157]:     b_blk,
        # src[scaled_mm.py:155-159]: ...
        accumulator = tl.dot(a_blk, b_blk, acc=accumulator, out_dtype=tl.float32)
    # src[scaled_mm.py:171]: c_blk = accumulator.to(out_dtype)
    v_0 = accumulator.to(c.type.element_ty)
    
    # src[scaled_mm.py:176]: c[tile_m, tile_n] = c_blk
    tl.store(c + (indices_0[:, None] * c_stride_0 + indices_1[None, :] * c_stride_1), v_0, mask_0[:, None] & mask_1[None, :])

def test_scaled_mm(a: torch.Tensor, b: torch.Tensor, scale_a: torch.Tensor, scale_b: torch.Tensor, out_dtype: torch.dtype, bias: torch.Tensor | None=None, *, _launcher=_default_launcher):
    # src[scaled_mm.py:125]: M, K = a.shape
    M, K = a.shape
    # src[scaled_mm.py:126]: N = b.shape[1]
    N = b.shape[1]
    # src[scaled_mm.py:134]: scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    # src[scaled_mm.py:135]: scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b
    # src[scaled_mm.py:147]: c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    # src[scaled_mm.py:148]: accumulator_dtype = torch.float32 if a.is_floating_point() else torch.int32
    accumulator_dtype = torch.float32 if a.is_floating_point() else torch.int32
    # src[scaled_mm.py:150]: for tile_m, tile_n in hl.tile([M, N]):
    _BLOCK_SIZE_1 = 128
    _BLOCK_SIZE_0 = 64
    # src[scaled_mm.py:150]: for tile_m, tile_n in hl.tile([M, N]):
    # src[scaled_mm.py:151]:     accumulator = hl.zeros([tile_m, tile_n], accumulator_dtype)
    # src[scaled_mm.py:152]:     for tile_k in hl.tile(K):
    # src[scaled_mm.py:150-176]: ...
    _launcher(_helion_scaled_mm, (triton.cdiv(N, _BLOCK_SIZE_1) * triton.cdiv(M, _BLOCK_SIZE_0),), a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), N, M, K, num_warps=2, num_stages=3)

    # _launcher(_helion_scaled_mm, (triton.cdiv(N, _BLOCK_SIZE_1) * triton.cdiv(M, _BLOCK_SIZE_0),), a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), N, M, K, num_warps=4, num_stages=2)

    # grid = lambda META: (
    #     triton.cdiv(M, _BLOCK_SIZE_0) * triton.cdiv(N, _BLOCK_SIZE_1),
    # )
    # _helion_scaled_mm[grid](a, b, c, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), N, M, K)
    # src[scaled_mm.py:178]: return c
    return c.to(out_dtype)


import copy
from dataclasses import dataclass

import torch
import triton

from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.scaled_mm import (
    generate_inputs,
    scaled_mm,
    baseline,
)

config_manager = ConfigManager()

@dataclass
class Row:
    case: str
    baseline_ms: float
    kernel_ms: float
    speedup_x: float

    baseline_peak_mb: float
    kernel_peak_mb: float
    mem_improve_x: float


def print_table(rows: list[Row]) -> None:
    headers = [
        "case",
        "baseline_ms",
        "kernel_ms",
        "speedup(x)",
        "baseline_peak(MB)",
        "kernel_peak(MB)",
        "mem_improve(x)",
    ]

    data = [
        [
            r.case,
            f"{r.baseline_ms:.3f}",
            f"{r.kernel_ms:.3f}",
            f"{r.speedup_x:.3f}",
            f"{r.baseline_peak_mb:.2f}",
            f"{r.kernel_peak_mb:.2f}",
            f"{r.mem_improve_x:.3f}",
        ]
        for r in rows
    ]

    cols = list(zip(*([headers] + data)))
    widths = [max(len(cell) for cell in col) for col in cols]

    def fmt(row: list[str]) -> str:
        return " | ".join(cell.ljust(w) for cell, w in zip(row, widths))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in data:
        print(fmt(row))


def cleanup_gpu_resources():
    import gc

    try:
        if torch.cuda.is_available():
            # Clear GPU memory cache
            torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            # Clear torch compilation cache
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()

            # Synchronize all CUDA streams
            torch.cuda.synchronize()

            # Reset peak memory stats for clean measurements
            torch.cuda.reset_peak_memory_stats()

            print("GPU resources cleaned up successfully")

    except Exception as e:
        print(f"Failed to cleanup GPU resources: {e}")


@torch.inference_mode()
def benchmark(fn, baseline, repeat=1000, cudagraph=True):
    rows: list[Row] = []
    benchmark_fn = (
        triton.testing.do_bench_cudagraph if cudagraph else triton.testing.do_bench
    )

    inputs_dict = generate_inputs()

    for key, inputs in inputs_dict.items():
        try:
            print(f"Start benchmarking with key {key}")

            inputs_clone = copy.deepcopy(inputs)

            helion_kernel = lambda: fn(*inputs)
            baseline_kernel = lambda: baseline(*inputs_clone)

            torch.cuda.reset_peak_memory_stats()
            helion_latency = benchmark_fn(helion_kernel, rep=repeat, return_mode="mean")
            helion_peak_mem = torch.cuda.max_memory_allocated() / 1e6

            torch.cuda.reset_peak_memory_stats()
            baseline_latency = benchmark_fn(
                baseline_kernel, rep=repeat, return_mode="mean"
            )
            baseline_peak_mem = torch.cuda.max_memory_allocated() / 1e6

            speedup = baseline_latency / helion_latency
            mem_improve = baseline_peak_mem / helion_peak_mem

            rows.append(
                Row(
                    case=key,
                    baseline_ms=baseline_latency,
                    kernel_ms=helion_latency,
                    speedup_x=speedup,
                    baseline_peak_mb=baseline_peak_mem,
                    kernel_peak_mb=helion_peak_mem,
                    mem_improve_x=mem_improve,
                )
            )

            cleanup_gpu_resources()

        except Exception as e:
            print(f"Benchmarking failed for key {key}: {e}")
            continue

    print_table(rows)


if __name__ == "__main__":
    benchmark(scaled_mm, test_scaled_mm)
