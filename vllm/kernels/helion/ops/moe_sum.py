# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helion implementation of ``ops.moe_sum``.

Reduces a fused MoE layer's per-expert outputs over the routing axis::

    output[t, :] = sum_k input[t, k, :]

When both ``topk_ids`` and ``expert_map`` are given, a lane this rank does not
compute contributes nothing -- its expert id is negative, or ``expert_map``
maps it to -1 -- which matches the ``torch.ops._moe_C.moe_sum`` pad-aware path.
With both missing, or with ``topk_ids`` alone, the reduction is plain:
``topk_ids`` on its own is ignored, where the C++ op would still drop its
negative lanes. ``expert_map`` without ``topk_ids`` is meaningless -- there is
no slot to index -- and is rejected. Callers pass the two together or (EP=1)
pass neither, and a single rank's ids are never off-rank, so the two agree on
every call production makes. The reduction runs in float32 and is cast back to
the output dtype.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.logger import init_logger
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
import helion.language as hl

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    # Tuned shapes follow the routed-layer shapes of the models vLLM serves;
    # the reduction is bandwidth bound, so a coarse num_tokens grid covers it.
    num_tokens_list = [32]
    hidden_size_list = [6144]
    top_k_list = [8]
    # The reduction reads routing two ways: ``topk_ids`` alone (skip lanes whose
    # expert id is negative) or with ``expert_map`` (also skip lanes off-rank for
    # this rank), or not at all. The three variants trace different programs --
    # 4, 3 and 2 loads/stores -- and helion sizes a config's structure lists
    # (``indexing``, eviction policies, ``range_*``) by that count: a longer
    # config is truncated onto a shorter program, a shorter one is rejected.
    # Tuning therefore uses the widest variant (EP > 1, below), whose config
    # serves all three, and the case key does not carry the variant.
    E = 64
    EP = 8
    assert E % EP == 0

    inputs: dict[CaseKey, tuple[Any, ...]] = {}
    for num_tokens, hidden_size, top_k in product(
        num_tokens_list, hidden_size_list, top_k_list
    ):
        input = torch.randn(
            num_tokens, top_k, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        output = torch.empty(
            num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        if EP == 1:
            # One rank owns every expert, so no lane is ever skipped and the
            # kernel takes its plain reduce.
            topk_ids = None
            expert_map = None
        else:
            topk_ids = torch.randint(
                0, E, (num_tokens, top_k), device="cuda", dtype=torch.int32
            )
            # Under EP this rank owns E // EP experts; every other expert maps
            # to -1 and its lanes are skipped. The draw is uniform, so which
            # rank's slice the input models does not matter -- and it has to
            # stay value-independent: the autotune script enumerates the case
            # keys under FakeTensorMode, where ops whose output shape depends on
            # tensor values (e.g. bincount) cannot run.
            num_local_experts = E // EP
            expert_map = torch.full((E,), -1, device="cuda", dtype=torch.int32)
            expert_map[:num_local_experts] = torch.arange(
                num_local_experts, device="cuda", dtype=torch.int32
            )
        key = CaseKey(
            {
                "num_tokens": num_tokens,
                "hidden_size": hidden_size,
                "top_k": top_k,
            }
        )
        inputs[key] = (input, output, topk_ids, expert_map)
    return inputs


_pick_cache: dict[tuple[int, int, int], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    The routing variant is not part of the key: the tuner always measures the
    widest one (topk_ids plus expert_map), and its config's structure lists
    are truncated onto the narrower programs, so one config serves all three.
    A config tuned for the same ``top_k`` is preferred -- ``top_k`` only bounds
    the reduction, so another one still applies -- then the closest
    ``hidden_size``, then the smallest ``num_tokens`` at least as large as the
    input's.
    """
    if not config_keys:
        return None

    input, _output, _topk_ids, _expert_map = args
    num_tokens, top_k, hidden_size = input.shape

    cache_key = (num_tokens, hidden_size, top_k)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    configs: list[CaseKey] = [key for key in config_keys if not key.is_default()]
    if not configs:
        return None
    exact_top_k = [key for key in configs if key["top_k"] == top_k]
    configs = exact_top_k or configs

    by_hidden: dict[int, list[CaseKey]] = {}
    for key in configs:
        by_hidden.setdefault(key["hidden_size"], []).append(key)

    best_hidden = min(by_hidden, key=lambda s: abs(s - hidden_size))
    candidates = sorted(by_hidden[best_hidden], key=lambda k: k["num_tokens"])
    chosen = next(
        (k for k in candidates if k["num_tokens"] >= num_tokens), candidates[-1]
    )
    result = CaseKey(dict(chosen))
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    input: torch.Tensor,  # [num_tokens, top_k, hidden_size]
    output: torch.Tensor,  # [num_tokens, hidden_size]
    topk_ids: torch.Tensor | None = None,  # [num_tokens, top_k] int32
    expert_map: torch.Tensor | None = None,  # [num_experts] int32, -1 = off-rank
) -> None:
    return


def baseline(
    input: torch.Tensor,
    output: torch.Tensor,
    topk_ids: torch.Tensor | None = None,
    expert_map: torch.Tensor | None = None,
) -> None:
    """Native-torch reference for the autotune accuracy check.

    A lane contributes when its expert id is non-negative and its expert is on
    this rank, which is only meaningful when both ``topk_ids`` and
    ``expert_map`` are given -- the same rule the kernel applies. The reduction
    runs in float32, like the kernel. See ``custom_baselines._moe_sum`` for the
    production-op comparison used by the benchmark.
    """
    x = input.to(torch.float32)
    if topk_ids is not None and expert_map is not None:
        keep = (topk_ids >= 0) & (expert_map[topk_ids.clamp(min=0)] >= 0)
        x = x * keep[:, :, None]
    output.copy_(x.sum(dim=1).to(output.dtype))


@register_kernel(
    mutates_args=["output"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        fast_math=True,
        autotune_baseline_fn=baseline,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def moe_sum(
    input: torch.Tensor,  # [num_tokens, top_k, hidden_size]
    output: torch.Tensor,  # [num_tokens, hidden_size] (out)
    topk_ids: torch.Tensor | None = None,  # [num_tokens, top_k] int32
    expert_map: torch.Tensor | None = None,  # [num_experts] int32, -1 = off-rank
) -> None:
    num_tokens, top_k, hidden_size = input.shape
    hl.specialize(hidden_size)
    assert output.shape == (num_tokens, hidden_size)
    assert output.is_contiguous()
    assert expert_map is None or topk_ids is not None, (
        "expert_map is only meaningful with topk_ids (the interleaved layout)"
    )

    # The grid-dependency wait is inlined right before the input load: the
    # routing tensors are produced by the routing stage, not by the immediately
    # preceding one, so the mask can be computed before waiting.
    for tile_m, tile_n in hl.tile([num_tokens, hidden_size]):
        acc = hl.zeros([tile_m, tile_n], dtype=torch.float32)
        for tile_k in hl.tile(top_k):
            # No ternary here: helion traces both arms, and the mask only
            # exists on the pad-aware path.
            if topk_ids is None or expert_map is None:
                # No routing to mask against: EP=1 hands over neither, and a
                # single rank's ids are never off-rank.
                hl.inline_triton(
                    "tl.extra.cuda.gdc_wait()", args=(), output_like=None
                )
                vals = input[tile_m, tile_k, tile_n]
            else:
                # A skipped lane is off-rank: its expert id is negative, or the
                # map sends it to -1. Its load is masked, and a masked lane
                # loads as zero, so it adds nothing.
                expert = topk_ids[tile_m, tile_k]
                mask = expert >= 0
                expert = hl.load(expert_map, [expert], extra_mask=mask)
                mask = mask & (expert >= 0)
                hl.inline_triton(
                    "tl.extra.cuda.gdc_wait()", args=(), output_like=None
                )
                vals = hl.load(
                    input, [tile_m, tile_k, tile_n], extra_mask=mask[:, :, None]
                )
            acc += vals.to(torch.float32).sum(dim=1)

        hl.store(output, [tile_m, tile_n], acc.to(output.dtype))

    hl.pdl_launch_dependents()
