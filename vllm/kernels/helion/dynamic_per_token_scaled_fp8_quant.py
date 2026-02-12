# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import helion
import helion.language as hl
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )


@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    allow_warp_specialize=True,
    autotune_baseline_atol=0.0,
    autotune_baseline_rtol=0.0,
    autotune_ignore_errors=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def dynamic_per_token_scaled_fp8_quant_v1(
    output: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    scale: torch.Tensor,  # [num_tokens, 1]
    scale_ub: torch.Tensor | None = None,  # scalar tensor
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)

    assert output.shape == input.shape
    assert scale.shape[0] == num_tokens
    assert scale.dtype == torch.float32
    assert input.stride()[-1] == 1
    assert output.stride()[-1] == 1

    fp8_min, fp8_max = get_fp8_min_max()
    min_scaling_factor = 1.0 / (fp8_max * 512.0)

    for tile_m in hl.tile(num_tokens):
        s_blk = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(dtype=torch.float32)
            tmp_blk = torch.amax(torch.abs(x_blk), dim=-1)
            s_blk = torch.maximum(s_blk, tmp_blk)

        if scale_ub is not None:
            scale_ub_s = hl.load(scale_ub, [])
            s_blk = s_blk.clamp(max=scale_ub_s)
        s_blk = s_blk * (1.0 / fp8_max)
        s_blk = s_blk.clamp(min=min_scaling_factor)
        scale[tile_m, 0] = s_blk

        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            y_blk = x_blk * (1.0 / s_blk[:, None])

            output[tile_m, tile_n] = y_blk.clamp(fp8_min, fp8_max).to(output.dtype)


@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    allow_warp_specialize=True,
    autotune_baseline_atol=0.0,
    autotune_baseline_rtol=0.0,
    autotune_ignore_errors=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def dynamic_per_token_scaled_fp8_quant_v2(
    output: torch.Tensor,  # [num_tokens, hidden_size]
    input: torch.Tensor,  # [num_tokens, hidden_size]
    scale: torch.Tensor,  # [num_tokens, 1]
    scale_ub: torch.Tensor | None = None,  # scalar tensor
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)

    assert output.shape == input.shape
    assert scale.shape[0] == num_tokens
    assert scale.dtype == torch.float32
    assert input.stride()[-1] == 1
    assert output.stride()[-1] == 1

    fp8_min, fp8_max = get_fp8_min_max()
    min_scaling_factor = 1.0 / (fp8_max * 512.0)

    for tile_m in hl.tile(num_tokens, block_size=1):
        s_blk = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(dtype=torch.float32)
            tmp_blk = torch.amax(torch.abs(x_blk), dim=-1)
            # hl.atomic_max(scale, [tile_m, 0], s_blk)
            # scale[tile_m, 0] = torch.maximum(scale[tile_m, 0], s_blk)
            s_blk = torch.maximum(s_blk, tmp_blk)

        if scale_ub is not None:
            scale_ub_s = hl.load(scale_ub, [])
            s_blk = s_blk.clamp(max=scale_ub_s)
        s_blk = s_blk * (1.0 / fp8_max)
        s_blk = s_blk.clamp(min=min_scaling_factor)
        scale[tile_m, 0] = s_blk

        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            y_blk = x_blk * (1.0 / s_blk[:, None])

            output[tile_m, tile_n] = y_blk.clamp(fp8_min, fp8_max).to(output.dtype)


from itertools import product
from pathlib import Path

from vllm.platforms import current_platform


def autotune(fn, force=False):
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    hidden_size_list = [2048, 4096, 8192]

    for num_tokens, hidden_size in product(num_tokens_list, hidden_size_list):
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
            input = torch.randn(
                num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
            )
            out_dtype: torch.dtype = current_platform.fp8_dtype()
            output = torch.empty(input.shape, device=input.device, dtype=out_dtype)
            scale = torch.empty(
                (num_tokens, 1), device=input.device, dtype=torch.float32
            )
            scale_ub = torch.mean(input).to(dtype=torch.float32, device=input.device)
            inputs = (output, input, scale, scale_ub)
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
    # num_tokens_list = [4096]
    hidden_size_list = [2048, 4096, 8192]
    # hidden_size_list = [4096]

    rows: list[Row] = []
    benchmark_fn = (
        triton.testing.do_bench_cudagraph if cudagraph else triton.testing.do_bench
    )

    for num_tokens, hidden_size in product(num_tokens_list, hidden_size_list):
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
            input = torch.randn(
                num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
            )
            out_dtype: torch.dtype = current_platform.fp8_dtype()
            output = torch.empty(input.shape, device=input.device, dtype=out_dtype)
            scale = torch.empty(
                (num_tokens, 1), device=input.device, dtype=torch.float32
            )
            scale_ub = torch.mean(input).to(dtype=torch.float32, device=input.device)
            inputs = (output, input, scale, scale_ub)
            fn.configs = [config]
            bound = fn.bind(inputs)
            compiled = bound.compile_config(config)

            def fake_impl(*args, **kwargs):
                return compiled(*args, **kwargs, _launcher=lambda *a, **kw: None)

            direct_register_custom_op(
                op_name=f"{fn.__name__}_{num_tokens}_{hidden_size}",
                op_func=fn,
                mutates_args=["output", "scale"],
                fake_impl=fake_impl,
                target_lib=vllm_helion_lib,
            )

            helion_custom_op = getattr(
                torch.ops.vllm_helion, f"{fn.__name__}_{num_tokens}_{hidden_size}"
            )
            helion_kernel = lambda: helion_custom_op(
                output.clone(), input, scale.clone(), scale_ub
            )
            baseline_kernel = lambda: baseline(
                output.clone(), input, scale.clone(), scale_ub
            )

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


if __name__ == "__main__":
    # autotune(dynamic_per_token_scaled_fp8_quant_v2, True)
    benchmark(dynamic_per_token_scaled_fp8_quant_v2, torch.ops._C.dynamic_per_token_scaled_fp8_quant)
