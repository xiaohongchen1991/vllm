# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import helion
import helion.language as hl
import torch

from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )


def is_weak_contiguous(x: torch.Tensor):
    strides = x.stride()
    sizes = x.shape
    is_not_transpose = strides[0] == 1 and (strides[1] >= max(1, sizes[0]))
    is_transpose = strides[1] == 1 and (strides[0] >= max(1, sizes[1]))
    return is_transpose or is_not_transpose


@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    allow_warp_specialize=True,
    autotune_baseline_atol=0.0,
    autotune_baseline_rtol=0.0,
    autotune_ignore_errors=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def scaled_mm(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    scale_b: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    M, K = a.shape
    N = b.shape[1]
    hl.specialize(K)
    hl.specialize(N)

    assert N > 0 and K > 0 and M > 0
    assert b.shape[0] == K
    assert a.dtype == b.dtype

    scale_a = scale_a.reshape(-1, 1) if scale_a.dim() <= 1 else scale_a
    scale_b = scale_b.reshape(-1, 1) if scale_b.dim() <= 1 else scale_b

    assert scale_a.dtype == scale_b.dtype and scale_a.is_floating_point()
    assert scale_a.shape[1] == 1 and (scale_a.shape[0] == 1 or scale_a.shape[0] == M)
    assert scale_b.shape[1] == 1 and (scale_b.shape[0] == 1 or scale_b.shape[0] == N)
    assert out_dtype.is_floating_point
    assert is_weak_contiguous(a)
    assert is_weak_contiguous(b)

    if bias is not None:
        assert bias.numel() == N and bias.dtype == out_dtype

    c = torch.empty((M, N), dtype=out_dtype, device=a.device)
    accumulator_dtype = torch.float32 if a.is_floating_point() else torch.int32

    for tile_m, tile_n in hl.tile([M, N]):
        accumulator = hl.zeros([tile_m, tile_n], accumulator_dtype)
        for tile_k in hl.tile(K):
            accumulator = hl.dot(
                a[tile_m, tile_k],
                b[tile_k, tile_n],
                acc=accumulator,
                out_dtype=accumulator_dtype,
            )

        scale_a_mask = (tile_m.index < scale_a.shape[0])[:, None]
        scale_a_blk = torch.where(scale_a_mask, scale_a[tile_m, :], scale_a[0, 0])
        accumulator = scale_a_blk * accumulator.to(torch.float32)

        scale_b_mask = (tile_n.index < scale_b.shape[0])[:, None]
        scale_b_blk = torch.where(scale_b_mask, scale_b[tile_n, :], scale_b[0, 0])
        accumulator = scale_b_blk.T * accumulator.to(torch.float32)

        c_blk = accumulator.to(out_dtype)

        if bias is not None:
            c_blk += bias[tile_n]

        c[tile_m, tile_n] = c_blk

    return c


from itertools import product
from pathlib import Path

from vllm.platforms import current_platform


def autotune(fn, force=False):
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    # num_tokens_list = [16]
    hidden_size_list = [2048, 4096, 8192]
    # hidden_size_list = [4096]

    for num_tokens, hidden_size in product(num_tokens_list, hidden_size_list):
        feature_size = hidden_size * 4
        try:
            print(
                f"Start autotuning with num_tokens={num_tokens} and hidden_size={hidden_size}"
            )
            path = Path(
                f"helion_configs/{fn.__name__}/num_tokens_{num_tokens}_hidden_size_{hidden_size}.json"
            )
            if not force and path.exists():
                print(
                    f"Config already exist. Skip autotuning with num_tokens={num_tokens} and hidden_size={hidden_size}"
                )
                continue
            in_dtype: torch.dtype = current_platform.fp8_dtype()
            out_dtype: torch.dtype = torch.bfloat16
            a = (
                0.25
                * torch.rand(
                    (num_tokens, hidden_size), dtype=torch.float32, device="cuda"
                )
            ).to(in_dtype)
            b = (
                0.25
                * torch.rand(
                    (feature_size, hidden_size), dtype=torch.float32, device="cuda"
                )
            ).to(in_dtype)
            b = b.t()
            scale_a = 0.25 * torch.rand((num_tokens, 1), device="cuda")
            scale_b = 0.25 * torch.rand((feature_size, 1), device="cuda")
            bias = torch.rand((feature_size,), device="cuda", dtype=out_dtype)
            inputs = (a, b, scale_a, scale_b, out_dtype, bias)
            fn.settings.autotune_effort = "full"
            best_config = fn.autotune(inputs, force=True)
            best_config.save(str(path))
            fn.reset()
            print(f"Successfully saved config file {str(path)}")
        except Exception as e:
            print(
                f"Autotuning failed with num_tokens={num_tokens} and hidden_size={hidden_size}: {e}"
            )
            continue


from dataclasses import dataclass

import triton
from torch.library import Library

from vllm.utils.torch_utils import direct_register_custom_op


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


vllm_helion_lib = Library("vllm_helion", "FRAGMENT")

def cleanup_gpu_resources():
    import gc

    try:
        if torch.cuda.is_available():
            # Clear GPU memory cache
            torch.cuda.empty_cache()

            # Force garbage collection
            gc.collect()

            # Clear torch compilation cache
            if hasattr(torch, '_dynamo'):
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
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    # num_tokens_list = [16]
    hidden_size_list = [2048, 4096, 8192]
    # hidden_size_list = [4096]

    rows: list[Row] = []
    benchmark_fn = (
        triton.testing.do_bench_cudagraph if cudagraph else triton.testing.do_bench
    )

    for num_tokens, hidden_size in product(num_tokens_list, hidden_size_list):
        feature_size = hidden_size * 4
        try:
            print(
                f"Start benchmarking with num_tokens={num_tokens} and hidden_size={hidden_size}"
            )
            path = Path(
                f"helion_configs/{fn.__name__}/num_tokens_{num_tokens}_hidden_size_{hidden_size}.json"
            )
            if not path.exists():
                print(
                    f"Config is missing. Skip benckmarking with num_tokens={num_tokens} and hidden_size={hidden_size}"
                )
                continue
            config = helion.Config.load(str(path))
            in_dtype: torch.dtype = current_platform.fp8_dtype()
            out_dtype: torch.dtype = torch.bfloat16
            a = (
                0.25
                * torch.rand(
                    (num_tokens, hidden_size), dtype=torch.float32, device="cuda"
                )
            ).to(in_dtype)
            b = (
                0.25
                * torch.rand(
                    (feature_size, hidden_size), dtype=torch.float32, device="cuda"
                )
            ).to(in_dtype)
            b = b.t()
            scale_a = 0.25 * torch.rand((num_tokens, 1), device="cuda")
            scale_b = 0.25 * torch.rand((feature_size, 1), device="cuda")
            bias = torch.rand((feature_size,), device="cuda", dtype=out_dtype)
            inputs = (a, b, scale_a, scale_b, out_dtype, bias)
            fn.configs = [config]
            bound = fn.bind(inputs)
            compiled = bound.compile_config(config)

            def fake_impl(*args, **kwargs):
                return compiled(*args, **kwargs, _launcher=lambda *a, **kw: None)

            direct_register_custom_op(
                op_name=f"{fn.__name__}_{num_tokens}_{hidden_size}",
                op_func=fn,
                mutates_args=None,
                fake_impl=fake_impl,
                target_lib=vllm_helion_lib,
            )

            helion_custom_op = getattr(
                torch.ops.vllm_helion, f"{fn.__name__}_{num_tokens}_{hidden_size}"
            )
            helion_kernel = lambda: helion_custom_op(*inputs)
            baseline_kernel = lambda: baseline(*inputs)

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
                    case=f"num_tokens_{num_tokens}_hidden_size_{hidden_size}",
                    baseline_ms=baseline_latency,
                    kernel_ms=helion_latency,
                    speedup_x=speedup,
                    baseline_peak_mb=baseline_peak_mem,
                    kernel_peak_mb=helion_peak_mem,
                    mem_improve_x=mem_improve,
                )
            )

            cleanup_gpu_resources()
            fn.reset()

        except Exception as e:
            print(
                f"Benchmarking failed with num_tokens={num_tokens} and hidden_size={hidden_size}: {e}"
            )
            continue

    print_table(rows)


def baseline(
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    scale_a: torch.Tensor,  # [1]/[1, 1]/[M]/[M, 1]
    scale_b: torch.Tensor,  # [1]/[1, 1]/[N]/[N, 1]
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,  # [N]
) -> torch.Tensor:
    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)
    return out


if __name__ == "__main__":
    # autotune(scaled_mm)
    benchmark(scaled_mm, baseline)
