# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from itertools import product
from typing import Any

import helion
import helion.language as hl
import regex as re
import torch

from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


@register_kernel(
    helion_settings=helion.Settings(
        autotune_ignore_errors=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def rms_norm(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor | None = None,
) -> None:
    x = input.view(-1, input.shape[-1])
    num_tokens, hidden_size = x.shape
    hl.specialize(hidden_size)

    assert input.dtype == weight.dtype
    assert weight.is_contiguous()

    y = output.view(-1, output.shape[-1])
    assert input.dtype == output.dtype
    assert y.size(0) == num_tokens
    assert y.size(1) == hidden_size

    if residual is not None:
        assert residual.dtype == input.dtype
        assert weight.is_contiguous()

    for tile_m in hl.tile(num_tokens, block_size=1):
        rms = hl.zeros([tile_m], dtype=torch.float32)
        for tile_n in hl.tile(hidden_size):
            x_blk = x[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
            rms = rms + x_blk.pow(2).sum(dim=-1)

        rms = torch.rsqrt(rms * (1.0 / hidden_size) + epsilon)

        for tile_n in hl.tile(hidden_size):
            x_blk = x[tile_m, tile_n].to(torch.float32)
            if residual is not None:
                x_blk = x_blk + residual[tile_m, tile_n]
            y_blk = (x_blk * rms[:, None]).to(input.dtype) * weight[None, tile_n]
            y[tile_m, tile_n] = y_blk
            # store residual after all uses of the old value are finished
            if residual is not None:
                residual[tile_m, tile_n] = x_blk


@rms_norm.register_input_generator  # type: ignore[misc]
def generate_inputs() -> dict[str, tuple[Any, ...]]:
    # TODO: it is difficult for kernel author to cover all input property combination.
    # Currently, dtypes are fixed. We need optimization to bucket/skip some combinations
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    hidden_size_list = [2048, 4096, 8192]
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = in_dtype
    inputs = {}

    for hidden_size, num_tokens in product(hidden_size_list, num_tokens_list):
        input = torch.randn(num_tokens, hidden_size, device="cuda", dtype=in_dtype)
        output = torch.empty(input.shape, device=input.device, dtype=out_dtype)
        residual = torch.randn_like(input)
        weight = torch.normal(
            mean=1.0,
            std=1.0,
            size=(hidden_size,),
            dtype=input.dtype,
            device=input.device,
        )
        epsilon = 1e-6

        config_key = f"hidden_size_{hidden_size}_num_tokens_{num_tokens}"
        inputs[config_key] = (
            output,
            input,
            weight,
            epsilon,
            residual,
        )

    return inputs


@rms_norm.register_config_picker  # type: ignore[misc]
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

    _, input, *_ = args
    hidden_size = input.size(-1)
    num_tokens = input.numel() // hidden_size

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


def rms_norm_baseline(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    residual: torch.Tensor | None = None,
) -> None:
    if residual is not None:
        output.copy_(input)
        torch.ops._C.fused_add_rms_norm(output, residual, weight, epsilon)
    else:
        torch.ops._C.rms_norm(output, input, weight, epsilon)
