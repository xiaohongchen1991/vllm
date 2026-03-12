# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import helion
import helion.language as hl
import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

def _get_fp8_dtype() -> torch.dtype:
    return current_platform.fp8_dtype()

def _get_int8_min_max() -> tuple[int, int]:
    qtype_traits = torch.iinfo(torch.int8)
    return qtype_traits.min, qtype_traits.max


def _get_int8_min_scaling_factor() -> float:
    return torch.finfo(torch.float32).eps

@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    allow_warp_specialize=True,
    autotune_baseline_atol=0.0,
    autotune_baseline_rtol=0.0,
    autotune_ignore_errors=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def rms_norm_per_block_quant(
    output: torch.Tensor, # [num_tokens, hidden_size]
    input: torch.Tensor, # [num_tokens, hidden_size]
    weight: torch.Tensor, # [hidden_size]
    scale: torch.Tensor, # [num_tokens, groups_per_row]
    epsilon: float,
    scale_ub: torch.Tensor, # []
    residual: torch.Tensor, # [num_tokens, hidden_size]
    group_size: int,
    is_scale_transposed: bool,
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, hidden_size = input.shape
    hl.specialize(hidden_size)
    hl.specialize(group_size)

    assert hidden_size % group_size == 0
    groups_per_row = hidden_size // group_size

    fp8_dtype = _get_fp8_dtype()
    assert output.dtype in [fp8_dtype, torch.int8]
    assert output.is_contiguous() and input.is_contiguous()

    if scale_ub is not None:
        assert output.dtype == fp8_dtype
        assert scale_ub.dtype == torch.float32

    assert input.dtype == weight.dtype
    assert scale.shape[0] == num_tokens
    assert scale.dtype == torch.float32

    if residual is not None:
        assert residual.dtype == input.dtype

    assert group_size in [64, 128]

    if scale.stride(1) > 1:
        assert is_scale_transposed

    quant_dtype = output.dtype
    qtype_traits_min: int | float
    qtype_traits_max: int | float
    if quant_dtype == torch.int8:
        qtype_traits_min, qtype_traits_max = _get_int8_min_max()
        min_scaling_factor = _get_int8_min_scaling_factor()
    else:
        qtype_traits_min, qtype_traits_max = get_fp8_min_max()
        min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)

    qtype_max = float(qtype_traits_max)

    for tile_m in hl.tile(num_tokens, block_size=1):
        rms = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            x_blk = input[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
            rms = rms + x_blk.pow(2).sum(dim=-1)

        rms = torch.rsqrt(rms * (1.0 / hidden_size) + epsilon)

        m_idx = tile_m.begin + hl.arange(tile_m.block_size)
        # shape: [tile_m, 1, 1]
        m_blk = m_idx[:, None, None]
        for tile_gn in hl.tile(groups_per_row):
            gn_idx = tile_gn.index
            n_offset = hl.arange(group_size)
            n_idx = gn_idx[:, None] * group_size + n_offset[None, :]
            # shape: [1, tile_gn, groups_per_row]
            n_blk = n_idx[None, :, :]
            mask = (gn_idx < groups_per_row)[None, :, None]

            # shape: [tile_m, tile_gn, groups_per_row]
            x_blk = hl.load(input, [m_blk, n_blk], extra_mask=mask).to(dtype=torch.float32)
            if residual is not None:
                r_blk = hl.load(residual, [m_blk, n_blk], extra_mask=mask)
                x_blk = x_blk + r_blk

            w_blk = hl.load(weight, [n_blk], extra_mask=mask)
            x_norm_blk = (x_blk * rms[:, None, None]).to(input.dtype) * w_blk
            s_blk = torch.amax(torch.abs(x_norm_blk), dim=-1).to(torch.float32)

            if scale_ub is not None:
                scale_ub_s = hl.load(scale_ub, [])
                s_blk = s_blk.clamp(max=scale_ub_s)

            s_blk = s_blk * (1.0 / qtype_max)
            s_blk = s_blk.clamp(min=min_scaling_factor)

            scale[tile_m, tile_gn] = s_blk

            if quant_dtype == torch.int8:
                y_blk = (x_norm_blk * (1.0 / s_blk[:, :, None])).round()
            else:
                y_blk = x_norm_blk / s_blk[:, :, None]

            y_blk = y_blk.clamp(qtype_traits_min, qtype_traits_max).to(
                output.dtype
            )
            hl.store(output, [m_blk, n_blk], y_blk, extra_mask=mask)

            # store residual after all uses of the old value are finished
            if residual is not None:
                hl.store(residual, [m_blk, n_blk], x_blk.to(residual.dtype), extra_mask=mask)

from itertools import product
from pathlib import Path


def autotune(fn, force=False):
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    # num_tokens_list = [16]
    hidden_size_list = [2048, 4096, 8192]
    # hidden_size_list = [2048]

    group_size = 128

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
                (num_tokens, hidden_size // group_size), device=input.device, dtype=torch.float32
            )
            scale_ub = torch.mean(input).to(dtype=torch.float32, device=input.device)
            residual = torch.randn_like(input)
            weight = torch.normal(
                mean=1.0,
                std=1.0,
                size=(hidden_size,),
                dtype=input.dtype,
                device=input.device,
            )
            epsilon = 1e-6
            inputs = (output, input, weight, scale, epsilon, scale_ub, residual, group_size, False)
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
    # hidden_size_list = [2048]

    group_size = 128

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
                (num_tokens, hidden_size // group_size), device=input.device, dtype=torch.float32
            )
            scale_ub = torch.mean(input).to(dtype=torch.float32, device=input.device)
            residual = torch.randn_like(input)
            weight = torch.normal(
                mean=1.0,
                std=1.0,
                size=(hidden_size,),
                dtype=input.dtype,
                device=input.device,
            )
            epsilon = 1e-6
            inputs = (output, input, weight, scale, epsilon, scale_ub, residual, group_size, False)
            fn.configs = [config]
            bound = fn.bind(inputs)
            compiled = bound.compile_config(config)

            def fake_impl(*args, **kwargs):
                return compiled(*args, **kwargs, _launcher=lambda *a, **kw: None)

            direct_register_custom_op(
                op_name=f"{fn.__name__}_{num_tokens}_{hidden_size}",
                op_func=fn,
                mutates_args=["output", "scale", "residual"],
                fake_impl=fake_impl,
                target_lib=vllm_helion_lib,
            )

            helion_custom_op = getattr(
                torch.ops.vllm_helion, f"{fn.__name__}_{num_tokens}_{hidden_size}"
            )
            helion_kernel = lambda: helion_custom_op(
                output.clone(),
                input,
                weight,
                scale.clone(),
                epsilon,
                scale_ub,
                residual.clone(),
                group_size,
                False,
            )
            baseline_kernel = lambda: baseline(
                output.clone(),
                input,
                weight,
                scale.clone(),
                epsilon,
                scale_ub,
                residual.clone(),
                group_size,
                False,
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
    autotune(rms_norm_per_block_quant, True)
    # benchmark(
    #     rms_norm_per_block_quant,
    #     torch.ops._C.rms_norm_per_block_quant,
    # )
