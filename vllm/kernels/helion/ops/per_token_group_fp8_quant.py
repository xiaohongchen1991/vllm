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


@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    allow_warp_specialize=True,
    autotune_baseline_atol=0.0,
    autotune_baseline_rtol=0.0,
    autotune_ignore_errors=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def per_token_group_fp8_quant(
    input: torch.Tensor,
    output_q: torch.Tensor,
    output_s: torch.Tensor,
    group_size: int,
    eps: float,
    fp8_min: float,
    fp8_max: float,
    scale_ue8m0: bool,
    dummy_is_scale_transposed: bool = False,
    dummy_is_tma_aligned: bool = False,
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)

    assert hidden_size % group_size == 0
    groups_per_row = hidden_size // group_size

    assert output_s.ndim == 2 and output_s.dtype == torch.float32

    for tile_m in hl.tile(num_tokens, block_size=1):
        m_idx = tile_m.begin + hl.arange(tile_m.block_size)
        m_blk = m_idx[:, None, None]
        for tile_gn in hl.tile(groups_per_row):
            gn_idx = tile_gn.index
            n_offset = hl.arange(group_size)
            n_idx = gn_idx[:, None] * group_size + n_offset[None, :]
            n_blk = n_idx[None, :, :]

            x_blk = input[m_blk, n_blk].to(dtype=torch.float32)
            y_s_blk = torch.clamp(torch.amax(torch.abs(x_blk), dim=-1), min=eps)
            y_s_blk = y_s_blk / fp8_max

            if scale_ue8m0:
                y_s_blk = torch.exp2(torch.ceil(torch.log2(y_s_blk)))

            y_q_blk = torch.clamp(x_blk / y_s_blk[:, :, None], fp8_min, fp8_max).to(output_q.dtype)

            output_s[tile_m, tile_gn] = y_s_blk
            output_q[m_blk, n_blk] = y_q_blk

from itertools import product
from pathlib import Path

from vllm.platforms import current_platform
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)

def autotune(fn, force=False):
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    hidden_size_list = [2048, 4096, 8192]
    # num_tokens_list = [16]
    # hidden_size_list = [4096]
    group_size = 128
    use_ue8m0 = False
    column_major = False
    fp8_min, fp8_max = get_fp8_min_max()
    eps = 1e-10

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
            output_q = torch.empty(input.shape, device=input.device, dtype=out_dtype)
            scale_shape = (num_tokens, hidden_size // group_size)
            if column_major:
                output_s = torch.empty_strided(scale_shape, (1, num_tokens), device=input.device, dtype=torch.float32)
            else:
                output_s = torch.empty(scale_shape, device=input.device, dtype=torch.float32)
            inputs = (input, output_q, output_s, group_size, eps, fp8_min, fp8_max, use_ue8m0)
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

    group_size = 128
    use_ue8m0 = False
    column_major = False
    fp8_min, fp8_max = get_fp8_min_max()
    eps = 1e-10

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
            output_q = torch.empty(input.shape, device=input.device, dtype=out_dtype)
            scale_shape = (num_tokens, hidden_size // group_size)
            if column_major:
                output_s = torch.empty_strided(scale_shape, (1, num_tokens), device=input.device, dtype=torch.float32)
            else:
                output_s = torch.empty(scale_shape, device=input.device, dtype=torch.float32)
            inputs = (input, output_q, output_s, group_size, eps, fp8_min, fp8_max, use_ue8m0)
            fn.configs = [config]
            bound = fn.bind(inputs)
            compiled = bound.compile_config(config)

            def fake_impl(*args, **kwargs):
                return compiled(*args, **kwargs, _launcher=lambda *a, **kw: None)

            direct_register_custom_op(
                op_name=f"{fn.__name__}_{num_tokens}_{hidden_size}",
                op_func=fn,
                mutates_args=["output_q", "output_s"],
                fake_impl=fake_impl,
                target_lib=vllm_helion_lib,
            )

            helion_custom_op = getattr(
                torch.ops.vllm_helion, f"{fn.__name__}_{num_tokens}_{hidden_size}"
            )
            helion_kernel = lambda: helion_custom_op(
                input, output_q.clone(), output_s.clone(), group_size, eps, fp8_min, fp8_max, use_ue8m0
            )
            baseline_kernel = lambda: baseline(
                input, output_q.clone(), output_s.clone(), group_size, eps, fp8_min, fp8_max, use_ue8m0, column_major, False
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
    # autotune(per_token_group_fp8_quant)
    benchmark(per_token_group_fp8_quant, torch.ops._C.per_token_group_fp8_quant)
