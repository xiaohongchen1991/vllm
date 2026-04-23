# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import product
from typing import Any

import helion
import helion.language as hl
import regex as re
import torch

from vllm.logger import init_logger
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

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)

def _get_fp8_dtype() -> torch.dtype:
    return current_platform.fp8_dtype()


def _get_int8_min_max() -> tuple[int, int]:
    qtype_traits = torch.iinfo(torch.int8)
    return qtype_traits.min, qtype_traits.max


def _get_int8_min_scaling_factor() -> float:
    return torch.finfo(torch.float32).eps

def generate_inputs() -> dict[str, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover all input
    # property combination. Currently, dtypes are fixed. We need optimization to
    # bucket/skip some combinations
    # num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    # intermediate_size = [6144, 12288, 28672]
    num_tokens_list = [32]
    intermediate_size_list = [12288]
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    inputs = {}
    for num_tokens, intermediate_size in product(num_tokens_list, intermediate_size_list):
        input = torch.randn(num_tokens, 2 * intermediate_size, device="cuda", dtype=in_dtype)
        result = torch.empty(num_tokens, intermediate_size, device=input.device, dtype=out_dtype)
        scale = torch.empty((num_tokens, 1), device=input.device, dtype=scale_dtype)
        scale_ub = torch.mean(input).to(scale_dtype)

        config_key = f"intermediate_size_{intermediate_size}_num_tokens_{num_tokens}"
        inputs[config_key] = (result, input, scale, scale_ub)

    return inputs


def pick_config(args: tuple[Any, ...], config_keys: list[str]) -> str | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest intermediate_size among available configs
         (exact match preferred).
      2. Among the num_tokens values tuned for that intermediate_size, pick
         the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.

    Config keys must be "default" or follow the format
    "intermediate_size_{int}_num_tokens_{int}".
    """

    if not config_keys:
        return None

    result, *_ = args
    num_tokens, intermediate_size = result.shape

    configs: dict[int, list[int]] = {}
    for key in config_keys:
        if key == "default":
            continue
        match = re.fullmatch(r"intermediate_size_(\d+)_num_tokens_(\d+)", key)
        if not match:
            raise ValueError(
                f"Malformed config key '{key}', "
                f"expected format 'intermediate_size_{{int}}_num_tokens_{{int}}'"
            )
        intermediate_size_str, num_tokens_str = match.groups()
        configs.setdefault(int(intermediate_size_str), []).append(int(num_tokens_str))

    if not configs:
        return "default" if "default" in config_keys else None

    best_intermediate_size = min(configs, key=lambda s: abs(s - intermediate_size))
    available_num_tokens = sorted(configs[best_intermediate_size])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens), available_num_tokens[-1]
    )

    return f"intermediate_size_{best_intermediate_size}_num_tokens_{best_num_tokens}"

def fake_impl(
    result: torch.Tensor,  # [num_tokens, intermediate_size]
    input: torch.Tensor,  # [num_tokens, 2 * intermediate_size]
    scale: torch.Tensor,  # [num_tokens, 1]
    scale_ub: torch.Tensor | None = None,  # scalar tensor
) -> None:
    return

@register_kernel(
    mutates_args=["result", "scale"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    # helion_settings=helion.Settings(
    #     autotune_baseline_fn=baseline,
    # ),
)  # type: ignore[misc]
def silu_and_mul_dynamic_per_token_quant(
    result: torch.Tensor,  # [num_tokens, intermediate_size]
    input: torch.Tensor,  # [num_tokens, 2 * intermediate_size]
    scale: torch.Tensor,  # [num_tokens, 1]
    scale_ub: torch.Tensor | None = None,  # scalar tensor
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, two_intermediate_size = input.shape
    hl.specialize(two_intermediate_size)

    assert two_intermediate_size % 2 == 0
    intermediate_size = two_intermediate_size // 2

    assert result.shape[0] == num_tokens
    assert result.shape[1] == intermediate_size

    assert scale.shape[0] == num_tokens
    assert scale.dtype == torch.float32
    assert input.stride()[-1] == 1
    assert result.stride()[-1] == 1

    fp8_min, fp8_max = get_fp8_min_max()
    min_scaling_factor = 1.0 / (fp8_max * 512.0)

    x_a = input[:, :intermediate_size]
    x_b = input[:, intermediate_size:]

    for tile_m in hl.tile(num_tokens, block_size=1):
        s_blk = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(intermediate_size):
            x_a_blk = x_a[tile_m, tile_n]
            x_b_blk = x_b[tile_m, tile_n]
            x_blk = torch.nn.functional.silu(x_a_blk.to(torch.float32)) * x_b_blk
            tmp_blk = torch.amax(torch.abs(x_blk), dim=-1)
            s_blk = torch.maximum(s_blk, tmp_blk)

        if scale_ub is not None:
            scale_ub_s = hl.load(scale_ub, [])
            s_blk = s_blk.clamp(max=scale_ub_s)
        s_blk = s_blk * (1.0 / fp8_max)
        s_blk = s_blk.clamp(min=min_scaling_factor)
        scale[tile_m, 0] = s_blk

        for tile_n in hl.tile(intermediate_size):
            x_a_blk = x_a[tile_m, tile_n]
            x_b_blk = x_b[tile_m, tile_n]
            x_blk = torch.nn.functional.silu(x_a_blk.to(torch.float32)) * x_b_blk
            y_blk = x_blk * (1.0 / s_blk[:, None])

            result[tile_m, tile_n] = y_blk.clamp(fp8_min, fp8_max).to(result.dtype)


# from vllm.model_executor.layers.activation import SiluAndMul
# from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
# from vllm.model_executor.layers.quantization.utils.quant_utils import (
#     GroupShape,
# )
# from vllm.config import VllmConfig, set_current_vllm_config
# import torch.nn as nn

# class Layer(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.fp8 = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

#     def forward(
#         self,
#         result: torch.Tensor,  # [num_tokens, intermediate_size]
#         input: torch.Tensor,  # [num_tokens, 2 * intermediate_size]
#         scale: torch.Tensor,  # [num_tokens, 1]
#         scale_ub: torch.Tensor | None = None,  # scalar tensor
#     ) -> tuple[torch.Tensor, torch.Tensor]:
#         act_result = SiluAndMul.forward_native(input)
#         result, scale = self.fp8.forward_native(act_result, None)
#         return result, scale

# config = VllmConfig()
# with set_current_vllm_config(config):
#     layer = Layer()
#     compiled_layer = torch.compile(layer.forward)

import vllm._custom_ops as ops
from vllm.model_executor.layers.activation import SiluAndMul

def baseline(
    result: torch.Tensor,  # [num_tokens, intermediate_size]
    input: torch.Tensor,  # [num_tokens, 2 * intermediate_size]
    scale: torch.Tensor,  # [num_tokens, 1]
    scale_ub: torch.Tensor | None = None,  # scalar tensor
) -> None:
    # compiled_layer(result, input, scale, scale_ub)
    silu_and_mul_out = SiluAndMul.forward_native(input)
    out, scale_out = ops.scaled_fp8_quant(silu_and_mul_out, scale=None, scale_ub=scale_ub, use_per_token_if_dynamic=True)
    result.copy_(out)
    scale.copy_(scale_out)
    
