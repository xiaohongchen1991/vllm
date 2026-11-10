# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helion silu-and-mul + per-block quantization of the MoE intermediate.

Same math as ``ops.silu_and_mul_per_block_quant`` -- ``silu(gate) * up``, then
one dynamic fp8 (or int8) scale per ``group_size`` columns -- with the two
changes the MoE path needs:

* ``clamp_limit``: the SwiGLU clamp applied *before* the nonlinearity (the gate
  clamped from above, the up half symmetrically), as ``swiglu_limit_func``
  does. The fused CUDA op has no clamp argument, so ``TritonExperts`` falls
  back to two launches; this kernel keeps it fused. ``None`` traces a variant
  without the clamp, so the common case carries no extra ops.
* ``num_valid``: the on-rank slot count, a device tensor (``None`` = every row
  is a candidate). Only rows below it hold slots this rank computes (the
  permuted MoE layouts pack them at the head), so the tile loop prunes the tail
  in-kernel and rows past the count are left untouched -- no host sync, so the
  composite stays CUDA-graph compatible. Layouts that interleave off-rank rows
  (the unpermuted staging is indexed by routed-slot id) pass ``topk_ids`` /
  ``expert_map`` instead, and rows are masked per slot exactly as
  ``ops.moe_sum`` decides them.

The standalone kernel's ``scale_ub`` (per-group amax upper bound) and
``is_scale_transposed`` (already a no-op placeholder) are not carried over:
the MoE block-quant path never bounds the scale, and the transposed layout
only exists for the standalone quantization kernels' callers.
"""

from itertools import product
from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.utils import (
    get_fp8_dtype,
    get_int8_min_max,
    get_int8_min_scaling_factor,
)
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

import helion
import helion.language as hl

from vllm.kernels.helion.register import register_kernel

logger = init_logger(__name__)


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    # TODO(xiaohongchen1991): it is difficult for kernel author to cover all input
    # property combination. Currently, dtypes are fixed. We need optimization to
    # bucket/skip some combinations
    num_tokens_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    intermediate_size_list = [6144, 12288, 25600]

    in_dtype: torch.dtype = torch.bfloat16
    out_dtype: torch.dtype = current_platform.fp8_dtype()
    scale_dtype: torch.dtype = torch.float32
    group_size_list = [128]
    inputs = {}
    for intermediate_size, group_size, num_tokens in product(
        intermediate_size_list, group_size_list, num_tokens_list
    ):
        input = torch.randn(
            num_tokens, 2 * intermediate_size, device="cuda", dtype=in_dtype
        )
        result = torch.empty(
            num_tokens, intermediate_size, device=input.device, dtype=out_dtype
        )
        scale = torch.empty(
            (num_tokens, intermediate_size // group_size),
            device=input.device,
            dtype=scale_dtype,
        )
        # The full row count: every row is a candidate, so the tuned config
        # covers the untrimmed grid and the runtime count can only shorten it.
        # No clamp and no per-slot routing: a SwiGLU limit and the routed-slot
        # mask are deployment properties, not tuning dimensions.
        num_valid = torch.full((1,), num_tokens, device=input.device, dtype=torch.int32)

        config_key = CaseKey(
            {
                "intermediate_size": intermediate_size,
                "group_size": group_size,
                "num_tokens": num_tokens,
            }
        )
        inputs[config_key] = (
            result,
            input,
            scale,
            group_size,
            None,
            num_valid,
            None,
            None,
        )

    return inputs


_pick_cache: dict[tuple[int, int, int], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    Selection strategy:
      1. Find the closest intermediate_size among available configs
         (exact match preferred).
      2. Find the closest group_size among available configs
         (exact match preferred).
      3. Among the num_tokens values tuned for that intermediate_size and group_size,
         pick the smallest num_tokens >= the input's num_tokens. If the input is
         larger than all available num_tokens, fall back to the largest.
    """

    if not config_keys:
        return None

    result, _, _, group_size, *_ = args
    num_tokens, intermediate_size = result.shape

    cache_key = (num_tokens, group_size, intermediate_size)
    cached = _pick_cache.get(cache_key)
    if cached is not None:
        return cached

    configs: dict[int, dict[int, list[int]]] = {}
    for key in config_keys:
        if key.is_default():
            continue
        configs.setdefault(key["intermediate_size"], {}).setdefault(
            key["group_size"], []
        ).append(key["num_tokens"])

    if not configs:
        return None

    best_intermediate_size = min(configs, key=lambda s: abs(s - intermediate_size))
    best_group_size = min(
        configs[best_intermediate_size], key=lambda s: abs(s - group_size)
    )
    available_num_tokens = sorted(configs[best_intermediate_size][best_group_size])
    best_num_tokens = next(
        (n for n in available_num_tokens if n >= num_tokens), available_num_tokens[-1]
    )

    result = CaseKey(
        {
            "intermediate_size": best_intermediate_size,
            "group_size": best_group_size,
            "num_tokens": best_num_tokens,
        }
    )
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    out: torch.Tensor,  # [num_tokens, intermediate_size]
    input: torch.Tensor,  # [num_tokens, 2 * intermediate_size]
    scales: torch.Tensor,  # [num_tokens, groups_per_row]
    group_size: int,
    clamp_limit: float | None = None,
    num_valid: torch.Tensor | None = None,  # [1] int32: on-rank slot count
    topk_ids: torch.Tensor | None = None,  # [rows, top_k] int32
    expert_map: torch.Tensor | None = None,  # [global_num_experts] int32
) -> None:
    return


def baseline(
    out: torch.Tensor,  # [num_tokens, intermediate_size]
    input: torch.Tensor,  # [num_tokens, 2 * intermediate_size]
    scales: torch.Tensor,  # [num_tokens, groups_per_row]
    group_size: int,
    clamp_limit: float | None = None,
    num_valid: torch.Tensor | None = None,  # [1] int32: on-rank slot count
    topk_ids: torch.Tensor | None = None,  # [rows, top_k] int32
    expert_map: torch.Tensor | None = None,  # [global_num_experts] int32
) -> None:
    num_tokens, intermediate_size = out.shape
    groups_per_row = intermediate_size // group_size
    quant_dtype = out.dtype
    qtype_min: int | float
    qtype_max: int | float

    if quant_dtype == torch.int8:
        qtype_min, qtype_max = get_int8_min_max()
        min_scaling_factor = get_int8_min_scaling_factor()
    else:
        qtype_min, qtype_max = get_fp8_min_max()
        min_scaling_factor = 1.0 / (qtype_max * 512.0)

    gate = input[:, :intermediate_size].to(torch.float32)
    up = input[:, intermediate_size:].to(torch.float32)
    if clamp_limit is not None:
        gate = gate.clamp(max=clamp_limit)
        up = up.clamp(min=-clamp_limit, max=clamp_limit)
    act = gate * torch.sigmoid(gate) * up
    x_grouped = act.view(num_tokens, groups_per_row, group_size)

    s = torch.amax(torch.abs(x_grouped), dim=-1)
    s = (s * (1.0 / qtype_max)).clamp(min=min_scaling_factor)

    y = x_grouped / s[:, :, None]
    if quant_dtype == torch.int8:
        y = y.round()

    # Rows past the padded count hold no on-rank slot; the kernel leaves them
    # untouched, so the reference blends the computed rows over them (a masked
    # copy, not a boolean-mask assignment, which would need a host sync).
    rows = torch.arange(num_tokens, device=out.device)
    keep = rows < (num_tokens if num_valid is None else num_valid[0])
    if topk_ids is not None:
        assert topk_ids.numel() == num_tokens
        slot_experts = topk_ids.reshape(-1).to(torch.long)
        on_rank = slot_experts >= 0
        if expert_map is not None:
            on_rank = on_rank & (expert_map[slot_experts.clamp(min=0)] >= 0)
        keep = keep & on_rank
    scales.copy_(torch.where(keep[:, None], s, scales))
    out.copy_(
        torch.where(
            keep[:, None],
            y.clamp(qtype_min, qtype_max).view(num_tokens, intermediate_size),
            out.to(torch.float32),
        ).to(out.dtype)
    )


@register_kernel(
    mutates_args=["out", "scales"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        fast_math=True,
        autotune_baseline_fn=baseline,
        ignore_warnings=[helion.exc.TensorOperationInWrapper],
    ),
)  # type: ignore[misc]
def moe_silu_and_mul_per_block_quant(
    out: torch.Tensor,  # [num_tokens, intermediate_size]
    input: torch.Tensor,  # [num_tokens, 2 * intermediate_size]
    scales: torch.Tensor,  # [num_tokens, groups_per_row]
    group_size: int,
    clamp_limit: float | None = None,  # SwiGLU clamp; None = no clamp
    num_valid: torch.Tensor | None = None,  # [1] int32: on-rank slot count
    topk_ids: torch.Tensor | None = None,  # [rows, top_k] int32
    expert_map: torch.Tensor | None = None,  # [global_num_experts] int32
) -> None:
    # This code assumes batch_dim and num_tokens are flattened
    assert input.ndim == 2
    num_tokens, two_intermediate_size = input.shape
    hl.specialize(two_intermediate_size)

    assert two_intermediate_size % 2 == 0
    intermediate_size = two_intermediate_size // 2

    assert out.shape[0] == num_tokens
    assert out.shape[1] == intermediate_size
    fp8_dtype = get_fp8_dtype()
    assert out.dtype in [fp8_dtype, torch.int8]

    if clamp_limit is not None:
        assert out.dtype == fp8_dtype
        assert clamp_limit > 0

    assert scales.ndim == 2 and scales.dtype == torch.float32

    assert scales.shape[0] == num_tokens
    groups_per_row = scales.shape[1]
    hl.specialize(groups_per_row)
    assert (
        intermediate_size % group_size == 0
        and intermediate_size // group_size == groups_per_row
    )

    assert group_size in [64, 128]
    hl.specialize(group_size)

    assert input.stride()[-1] == 1
    assert out.stride()[-1] == 1

    # The three layouts are mutually exclusive: the counted compact layout
    # (``num_valid``), the gathered routed-slot layout without EP
    # (``topk_ids`` only; every slot is on-rank), and the gathered layout with
    # EP (``topk_ids`` + ``expert_map``). ``num_valid`` describes the compact
    # prefix, where every row below the count is on-rank, so combining it with
    # per-slot masking would mean the two disagree about the layout.
    assert expert_map is None or topk_ids is not None, (
        "expert_map is only meaningful with topk_ids (the interleaved layout)"
    )
    assert num_valid is None or (topk_ids is None and expert_map is None), (
        "num_valid describes the compact prefix of the permuted layout; it "
        "cannot be combined with per-slot topk_ids/expert_map masking"
    )

    quant_dtype = out.dtype
    qtype_traits_min: int | float
    qtype_traits_max: int | float
    if quant_dtype == torch.int8:
        qtype_traits_min, qtype_traits_max = get_int8_min_max()
        min_scaling_factor = get_int8_min_scaling_factor()
    else:
        qtype_traits_min, qtype_traits_max = get_fp8_min_max()
        min_scaling_factor = 1.0 / (qtype_traits_max * 512.0)

    qtype_max = float(qtype_traits_max)

    input = input.view(num_tokens, -1, group_size)
    out = out.view(num_tokens, -1, group_size)

    if topk_ids is not None:
        flat_topk = topk_ids.reshape(-1)

    hl.pdl_wait()
    hl.pdl_launch_dependents()

    for tile_m, tile_gn, tile_n in hl.tile(
        [num_tokens, groups_per_row, group_size], block_size=[None, None, group_size]
    ):
        # Rows past the on-rank slot count hold no data and keep whatever they
        # held (the composite stages this buffer, so rows no GEMM reads must not
        # be written). Without a count every staged row is a candidate: the
        # bound becomes the row extent itself. Off-rank rows exist only in the
        # interleaved routed-slot layout with EP (topk_ids + expert_map), which
        # masks them per slot below; the compact counted layout has none.
        # The count is loaded here, inside the loop, where a scalar subscript
        # lowers to a load (a host-side tensor value cannot be used directly).
        if num_valid is None:  # noqa: SIM108 - helion traces both ternary arms
            ntp = num_tokens
        else:
            ntp = num_valid[0]
        if tile_m.begin < ntp:
            if topk_ids is not None and expert_map is not None and num_valid is None:
                # Interleaved routed-slot layout with EP: validity per slot, as
                # ``ops.moe_sum`` decides it -- the row is a routed slot, so it
                # is on-rank iff its expert maps here. ``num_valid is None`` is
                # asserted above for every per-slot call: the counted compact
                # layout never needs this mask.
                e = hl.load(flat_topk, [tile_m])
                row_mask = e >= 0
                mapped = hl.load(expert_map, [e], extra_mask=row_mask)
                row_mask = row_mask & (mapped >= 0)
            else:
                # Counted layout: the prefix guard above is the whole test (a
                # constant-true mask keeps one shape for the masks below). This
                # covers the compact permuted layout (num_valid set) and the
                # gathered layout without EP (expert_map is None: every slot is
                # on-rank, so no per-slot masking is needed).
                row_mask = tile_m.index >= 0
            col_mask = (tile_gn.index + groups_per_row < 2 * groups_per_row)[
                None, :, None
            ]
            x_a_blk = hl.load(
                input,
                [tile_m, tile_gn, tile_n],
                extra_mask=row_mask[:, None, None],
            ).to(torch.float32)
            x_b_blk = hl.load(
                input,
                [tile_m, tile_gn.index + groups_per_row, tile_n],
                extra_mask=col_mask & row_mask[:, None, None],
            ).to(torch.float32)
            if clamp_limit is not None:
                # SwiGLU clamp, as swiglu_limit_func applies it: the gate from
                # above only, the up half symmetrically, before the nonlinearity.
                x_a_blk = torch.clamp(x_a_blk, max=clamp_limit)
                x_b_blk = torch.clamp(x_b_blk, min=-clamp_limit, max=clamp_limit)
            x_blk = x_a_blk * torch.sigmoid(x_a_blk) * x_b_blk
            s_blk = torch.amax(torch.abs(x_blk), dim=-1).to(torch.float32)
            s_blk = s_blk * (1.0 / qtype_max)
            s_blk = s_blk.clamp(min=min_scaling_factor)

            hl.store(
                scales,
                [tile_m, tile_gn],
                s_blk,
                extra_mask=row_mask[:, None],
            )
            if quant_dtype == torch.int8:
                y_blk = (x_blk * (1.0 / s_blk[:, :, None])).round()
            else:
                y_blk = x_blk / s_blk[:, :, None]

            hl.store(
                out,
                [tile_m, tile_gn, tile_n],
                y_blk.clamp(qtype_traits_min, qtype_traits_max).to(out.dtype),
                extra_mask=row_mask[:, None, None],
            )
