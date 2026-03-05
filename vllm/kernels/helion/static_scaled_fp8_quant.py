# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import helion
import helion.language as hl
import torch
from typing import Callable, Any, Optional, Sequence

@helion.kernel(
    autotune_effort="none",
    static_shapes=False,
    allow_warp_specialize=True,
    autotune_baseline_atol=0.0,
    autotune_baseline_rtol=0.0,
    autotune_ignore_errors=True,
    ignore_warnings=[helion.exc.TensorOperationInWrapper],
)
def static_scaled_fp8_quant_helion(
    output: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    group_m: int,
    group_n: int,
) -> None:
    assert scale.dim() == 2
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2

    num_tokens, hidden_size = input.shape
    num_groups_m, num_groups_n = scale.shape
    hl.specialize(hidden_size)
    hl.specialize(group_n)
    hl.specialize(num_groups_n)

    for tile_gm, tile_gn, tile_m, tile_n in hl.tile(
        [num_groups_m, num_groups_n, group_m, group_n]
    ):
        gm_idx = tile_gm.index
        gn_idx = tile_gn.index

        # shape: [tile_gm, tile_gn]
        scale_blk = scale[gm_idx[:, None], gn_idx[None, :]]
        inv_scale_blk = (1.0 / scale_blk).to(dtype=torch.float32)

        # offset inside group
        m_offset = tile_m.index
        n_offset = tile_n.index

        # Global indices
        # m_idx: [tile_gm, tile_m]
        # n_idx: [tile_gn, tile_n]
        m_idx = gm_idx[:, None] * group_m + m_offset[None, :]
        n_idx = gn_idx[:, None] * group_n + n_offset[None, :]

        m_blk = m_idx[:, None, :, None]
        n_blk = n_idx[None, :, None, :]

        # input tile shape:  [tile_gm, tile_gn, tile_m, tile_n]
        x_blk = input[m_blk, n_blk].to(torch.float32)

        # scale tile shape:  [tile_gm, tile_gn, 1, 1]
        y_blk = x_blk * inv_scale_blk[:, :, None, None]

        output[m_blk, n_blk] = y_blk.to(output.dtype)


def static_scaled_fp8_quant(
    output: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    group_shape: Optional[Sequence[int]] = None,
    fn: Callable[..., None] = static_scaled_fp8_quant_helion,
) -> None:
    assert input.stride(-1) == 1
    assert output.stride(-1) == 1

    num_tokens, hidden_size = input.shape

    if scale.dim() == 0 or scale.numel() == 1:
        # per tensor
        group_m = num_tokens
        group_n = hidden_size
        scale_2d = scale.reshape(1, 1)
    elif scale.dim() == 1:
        assert group_shape is not None and len(group_shape) == 2
        group_shape_m, group_shape_n = group_shape
        assert group_shape_m == -1 or group_shape_n == -1
        group_m = num_tokens if group_shape_m == -1 else int(group_shape_m)
        group_n = hidden_size if group_shape_n == -1 else int(group_shape_n)
        scale_2d = scale[None, :] if group_shape_m == -1 else scale[:, None]
        inferred_group_m = num_tokens // scale_2d.size(0)
        inferred_group_n = hidden_size // scale_2d.size(1)
        assert group_m == inferred_group_m and group_n == inferred_group_n
    elif scale.dim() == 2:
        scale_2d = scale
        scale_size_0, scale_size_1 = scale.shape
        assert num_tokens % scale_size_0 == 0
        assert hidden_size % scale_size_1 == 0
        inferred_group_m = num_tokens // scale_size_0
        inferred_group_n = hidden_size // scale_size_1

        if group_shape is not None:
            assert len(group_shape) == 2
            group_shape_m, group_shape_n = group_shape
            group_m = num_tokens if group_shape_m == -1 else int(group_shape_m)
            group_n = hidden_size if group_shape_n == -1 else int(group_shape_n)
            assert group_m == inferred_group_m and group_n == inferred_group_n
        else:
            group_m, group_n = inferred_group_m, inferred_group_n

    fn(output, input, scale_2d, group_m, group_n)


from itertools import product
from pathlib import Path

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    scaled_quantize,
)
from vllm.platforms import current_platform


def autotune(fn, force=False):
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    # num_tokens_list = [16]
    hidden_size_list = [2048, 4096, 8192]
    # hidden_size_list = [4096]
    group_shape = (-1, 1)

    for num_tokens, hidden_size in product(num_tokens_list, hidden_size_list):
        try:
            print(
                f"Start autotuning with num_tokens={num_tokens} and hidden_size={hidden_size}"
            )
            path = Path(
                f"helion_configs/{fn.__name__}/num_tokens_{num_tokens}_hidden_size_{hidden_size}_group_shape_{group_shape}.json"
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
            _, scale = scaled_quantize(
                input,
                group_shape,
                current_platform.fp8_dtype(),
                compute_dtype=torch.float32,
            )
            if scale.dim() == 0 or scale.numel() == 1:
                group_m = num_tokens
                group_n = hidden_size
                scale_2d = scale.reshape(1, 1)
            elif scale.dim() == 1:
                group_shape_m, group_shape_n = group_shape
                group_m = num_tokens if group_shape_m == -1 else int(group_shape_m)
                group_n = hidden_size if group_shape_n == -1 else int(group_shape_n)
                scale_2d = scale[None, :] if group_shape_m == -1 else scale[:, None]
            elif scale.dim() == 2:
                scale_2d = scale
                group_shape_m, group_shape_n = group_shape
                group_m = num_tokens if group_shape_m == -1 else int(group_shape_m)
                group_n = hidden_size if group_shape_n == -1 else int(group_shape_n)

            inputs = (output, input, scale_2d, group_m, group_n)
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
    group_shape = (-1, 1)

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
                f"helion_configs/{fn.__name__}/num_tokens_{num_tokens}_hidden_size_{hidden_size}_group_shape_{group_shape}.json"
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
            _, scale = scaled_quantize(
                input,
                group_shape,
                current_platform.fp8_dtype(),
                compute_dtype=torch.float32,
            )
            if scale.dim() == 0 or scale.numel() == 1:
                group_m = num_tokens
                group_n = hidden_size
                scale_2d = scale.reshape(1, 1)
            elif scale.dim() == 1:
                group_shape_m, group_shape_n = group_shape
                group_m = num_tokens if group_shape_m == -1 else int(group_shape_m)
                group_n = hidden_size if group_shape_n == -1 else int(group_shape_n)
                scale_2d = scale[None, :] if group_shape_m == -1 else scale[:, None]
            elif scale.dim() == 2:
                scale_2d = scale
                group_shape_m, group_shape_n = group_shape
                group_m = num_tokens if group_shape_m == -1 else int(group_shape_m)
                group_n = hidden_size if group_shape_n == -1 else int(group_shape_n)

            inputs = (output, input, scale_2d, group_m, group_n)
            fn.configs = [config]
            bound = fn.bind(inputs)
            compiled = bound.compile_config(config)

            def fake_impl(*args, **kwargs):
                return compiled(*args, **kwargs, _launcher=lambda *a, **kw: None)

            direct_register_custom_op(
                op_name=f"{fn.__name__}_{num_tokens}_{hidden_size}",
                op_func=fn,
                mutates_args=["output"],
                fake_impl=fake_impl,
                target_lib=vllm_helion_lib,
            )

            helion_custom_op = getattr(
                torch.ops.vllm_helion, f"{fn.__name__}_{num_tokens}_{hidden_size}"
            )
            helion_kernel = lambda: helion_custom_op(
                output.clone(), input, scale_2d, group_m, group_n
            )
            baseline_kernel = lambda: baseline(
                output.clone(), input, scale, group_shape
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

def key_fn(
    output: torch.Tensor,
    input: torch.Tensor,
    scale: torch.Tensor,
    group_m: int,
    group_n: int,
):
    try:
        hash(input.shape)
        print("input shape: ", input.shape)
        num_tokens, hidden_size = input.shape
        return (helion.next_power_of_2(num_tokens), helion.next_power_of_2(hidden_size))
    except:
        return (0, 0)

_REGISTERED = False
def register_kernel() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    fn = static_scaled_fp8_quant_helion
    fn.configs = [
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_1_hidden_size_2048_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_2_hidden_size_2048_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_4_hidden_size_2048_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_8_hidden_size_2048_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_16_hidden_size_2048_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_32_hidden_size_2048_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_64_hidden_size_2048_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_128_hidden_size_2048_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_256_hidden_size_2048_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_512_hidden_size_2048_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_1024_hidden_size_2048_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_2048_hidden_size_2048_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_4096_hidden_size_2048_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_1_hidden_size_8192_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_2_hidden_size_8192_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_4_hidden_size_8192_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_8_hidden_size_8192_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_16_hidden_size_8192_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_32_hidden_size_8192_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_64_hidden_size_8192_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_128_hidden_size_8192_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_256_hidden_size_8192_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_512_hidden_size_8192_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_1024_hidden_size_8192_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_2048_hidden_size_8192_group_shape_(-1, -1).json"))),
        helion.Config.load(str(Path(f"helion_configs/{fn.__name__}/num_tokens_4096_hidden_size_8192_group_shape_(-1, -1).json"))),
    ]
    fn._key_fn = key_fn

    def fn_wrapper(
        output: torch.Tensor,
        input: torch.Tensor,
        scale: torch.Tensor,
        group_shape: Optional[Sequence[int]]
    ) -> None:
        return static_scaled_fp8_quant(output, input, scale, group_shape, fn)

    def fake_impl(*args, **kwargs):
        return

    direct_register_custom_op(
        op_name="static_scaled_fp8_quant",
        op_func=fn_wrapper,
        mutates_args=["output"],
        fake_impl=fake_impl,
        target_lib=vllm_helion_lib,
    )
    _REGISTERED = True
    print("successfully static_scaled_fp8_quant")

if __name__ == "__main__":
    # autotune(static_scaled_fp8_quant_helion, True)
    benchmark(static_scaled_fp8_quant_helion, torch.ops._C.static_scaled_fp8_quant)
