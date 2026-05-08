import copy
from dataclasses import dataclass

import torch
import triton

from vllm.kernels.helion.config_manager import ConfigManager
from vllm.kernels.helion.ops.per_token_group_fp8_quant import (
    generate_inputs,
    helion_kernel,
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
        # "baseline_peak(MB)",
        # "kernel_peak(MB)",
        # "mem_improve(x)",
    ]

    data = [
        [
            r.case,
            f"{r.baseline_ms:.3f}",
            f"{r.kernel_ms:.3f}",
            f"{r.speedup_x:.3f}",
            # f"{r.baseline_peak_mb:.2f}",
            # f"{r.kernel_peak_mb:.2f}",
            # f"{r.mem_improve_x:.3f}",
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

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            peak_mem_before = torch.cuda.max_memory_allocated()
            helion_latency = benchmark_fn(helion_kernel, rep=repeat, return_mode="mean")
            torch.cuda.synchronize()
            peak_mem_after = torch.cuda.max_memory_allocated()
            helion_peak_mem = (peak_mem_after - peak_mem_before) / 1e6

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            peak_mem_before = torch.cuda.max_memory_allocated()
            baseline_latency = benchmark_fn(
                baseline_kernel, rep=repeat, return_mode="mean"
            )
            torch.cuda.synchronize()
            peak_mem_after = torch.cuda.max_memory_allocated()
            baseline_peak_mem = (peak_mem_after - peak_mem_before) / 1e6

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
            raise e
            print(f"Benchmarking failed for key {key}: {e}")
            continue

    print_table(rows)


if __name__ == "__main__":
    benchmark(helion_kernel, baseline)
