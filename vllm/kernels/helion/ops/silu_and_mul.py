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

def generate_inputs() -> dict[str, tuple[Any, ...]]:
    # TODO: it is difficult for kernel author to cover all input property combination.
    # Currently, dtypes are fixed. We need optimization to bucket/skip some combinations
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    intermediate_size_list = [2048, 2880, 4096, 8192, 11008, 14336]
    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = in_dtype
    inputs = {}

    for intermediate_size, num_tokens in product(
        intermediate_size_list, num_tokens_list
    ):
        input = torch.randn(
            num_tokens,
            2 * intermediate_size,
            device="cuda",
            dtype=in_dtype,
        )
        result = torch.empty(
            (num_tokens, intermediate_size), device=input.device, dtype=out_dtype
        )

        config_key = f"intermediate_size_{intermediate_size}_num_tokens_{num_tokens}"
        inputs[config_key] = (
            result,
            input,
        )

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

    result, _ = args
    intermediate_size = result.size(-1)
    num_tokens = result.numel() // intermediate_size

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
    result: torch.Tensor,
    input: torch.Tensor
) -> None:
    return


@register_kernel(
    mutates_args=["result"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_ignore_errors=True,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def silu_and_mul(result: torch.Tensor, input: torch.Tensor) -> None:
    x = input.view(-1, input.shape[-1])
    num_tokens, two_intermediate_size = x.shape
    hl.specialize(two_intermediate_size)
    assert two_intermediate_size % 2 == 0
    intermediate_size = two_intermediate_size // 2

    y = result.view(-1, result.shape[-1])
    assert y.shape[0] == num_tokens
    assert y.shape[1] == intermediate_size

    x_a = x[:, :intermediate_size]
    x_b = x[:, intermediate_size:]

    for tile_m, tile_n in hl.tile([num_tokens, intermediate_size]):
        x_a_blk = x_a[tile_m, tile_n]
        x_b_blk = x_b[tile_m, tile_n]
        y[tile_m, tile_n] = (
            torch.nn.functional.silu(x_a_blk.to(torch.float32)) * x_b_blk
        )


def silu_and_mul_baseline(result: torch.Tensor, input: torch.Tensor) -> None:
    torch.ops._C.silu_and_mul(result, input)
