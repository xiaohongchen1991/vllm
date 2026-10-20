# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from itertools import product
from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import _ceil_to_ue8m0
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
import helion.language as hl
from helion._testing import assert_close_with_mismatch_tolerance
from helion.autotuner import BooleanFragment, PowerOfTwoFragment

from vllm.kernels.helion.register import register_kernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.utils import _resize_cache

logger = init_logger(__name__)

GROUP_K = 128
GROUP_N = 128


def _simulate_topk_ids(num_tokens: int, E: int, top_k: int) -> torch.Tensor:
    """Per-token top_k expert ids from a simulated uniform router.

    Mirrors ``simulate_expert_counts``: Gaussian gating logits, top_k selection
    (distinct experts per token), seeded per shape for reproducibility. Runs
    eagerly so it produces concrete ids under the autotuner's FakeTensorMode.
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    with unset_fake_temporarily():
        gen = torch.Generator(device="cuda").manual_seed(
            num_tokens * 1000 + E * 10 + top_k
        )
        gating = torch.randn(num_tokens, E, device="cuda", generator=gen)
        return gating.topk(top_k, dim=-1).indices.to(torch.int32)


def _generate_input(
    E: int,
    K: int,
    N: int,
    num_tokens: int,
    top_k: int,
    fp8: torch.dtype,
) -> tuple[Any, ...]:
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    scale_dtype = torch.float32
    out_dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(K)

    num_group_k = K // GROUP_K
    num_group_n = N // GROUP_N

    # GEMM1 projects to the gate_up tensor [*, N]; silu_and_mul folds it to the
    # [*, N // 2] intermediate; GEMM2 (down projection) maps the intermediate
    # back to the model hidden dim H == K, producing the layer's [num_tokens,
    # H] output (the top_k-weighted rows reduced over top_k).
    intermediate = N // 2
    H = K
    num_group_i = intermediate // GROUP_N
    num_group_h = H // GROUP_N

    topk_ids = _simulate_topk_ids(num_tokens, E, top_k)

    # Routed combine weights, seeded per shape like the ids.
    with unset_fake_temporarily():
        gen = torch.Generator(device="cuda").manual_seed(
            num_tokens * 1000 + top_k * 10 + 7
        )
        topk_weights = torch.rand(
            num_tokens, top_k, dtype=scale_dtype, device="cuda", generator=gen
        )

    scale2 = 1.0 / math.sqrt(intermediate)
    hidden_states = (scale * (0.5 + torch.rand(num_tokens, K, device="cuda"))).to(fp8)
    w1 = (scale * (0.5 + torch.rand(E, N, K, device="cuda"))).to(fp8)
    w2 = (scale2 * (0.5 + torch.rand(E, H, intermediate, device="cuda"))).to(fp8)
    # Layer-level output: the top_k-weighted rows reduced over top_k.
    output = torch.zeros((num_tokens, H), dtype=out_dtype, device="cuda")

    a1q_scale = 0.5 + torch.rand(
        num_group_k, num_tokens, dtype=scale_dtype, device="cuda"
    )
    a1q_scale = _ceil_to_ue8m0(a1q_scale).t()

    # Weight scales: w1 [E, num_group_n, num_group_k], w2 [E, num_group_h,
    # num_group_i].
    w1_scale = 0.5 + torch.rand(
        E, num_group_n, num_group_k, dtype=scale_dtype, device="cuda"
    )
    w1_scale = _ceil_to_ue8m0(w1_scale)
    w2_scale = 0.5 + torch.rand(
        E, num_group_h, num_group_i, dtype=scale_dtype, device="cuda"
    )
    w2_scale = _ceil_to_ue8m0(w2_scale)

    expert_map = None
    global_num_experts = E

    # Scratch workspaces shared by the Helion kernel and the custom baselines
    # (the same tuple is forwarded to both). The Helion kernel stages the fp8
    # intermediate and its f32 scales in workspace13 (byte partitioned: a2q
    # rows from the base, the scales just past them) and uses workspace2 for
    # the bf16 gate_up cache [num_tokens, top_k, N] and, past it, the bf16
    # GEMM2 output [num_tokens, top_k, H]; the Triton baseline stages gate_up
    # in workspace2 (first N slice) and the weighted GEMM2 output in
    # workspace13.
    workspace13 = torch.empty(
        (num_tokens, top_k, max(intermediate, H)), dtype=out_dtype, device="cuda"
    )
    workspace2 = torch.empty((num_tokens, top_k, N + H), dtype=out_dtype, device="cuda")
    return (
        output,
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        a1q_scale,
        topk_weights,
        topk_ids,
        expert_map,
        global_num_experts,
        workspace13,
        workspace2,
    )


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    # The Helion kernel is autotuned per shape.
    # num_tokens_list follows the cudagraph_capture_sizes pattern but is capped
    # here to cover only small values.
    num_tokens_list = [1, 2, 4, 8, 16, 24, 32]

    fp8: torch.dtype = current_platform.fp8_dtype()

    # (E, top_k, (K, N)) entries.
    e_topk_kn_list: list[tuple[int, int, tuple[int, int]]] = [
        (256, 8, (2048, 1024)),
        # (256, 8, (512, 2048)),
    ]

    inputs = {}
    for num_tokens, (E, top_k, (K, N)) in product(num_tokens_list, e_topk_kn_list):
        inputs[
            CaseKey(
                {
                    "E": E,
                    "K": K,
                    "N": N,
                    "topk": top_k,
                    "num_tokens": num_tokens,
                    "in_dtype": str(fp8),
                }
            )
        ] = _generate_input(E, K, N, num_tokens, top_k, fp8)

    return inputs


_pick_cache: dict[tuple[int, int, int, int, int, str], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config for the given input shape.

    Configs are matched within the runtime input dtype and ``top_k``. E/K/N are
    picked by closest match. ``num_tokens`` (passed by the caller) is bucketed to
    the smallest tuned value >= runtime.
    """

    if not config_keys:
        return None

    (
        output,
        hidden_states,
        w1,
        w2,
        w1_s,
        w2_s,
        a1q_s,
        topk_w,
        topk_ids,
        expert_map,
        gne,
        *_,
    ) = args

    num_tokens, K = hidden_states.shape
    top_k = topk_ids.shape[1]
    E, N = w1.shape[0], w1.shape[1]
    in_dtype = str(hidden_states.dtype)

    cache_key = (E, K, N, top_k, num_tokens, in_dtype)
    if cache_key in _pick_cache:
        return _pick_cache[cache_key]

    candidates: list[CaseKey] = []
    for key in config_keys:
        if key.is_default():
            continue
        if not all(k in key for k in ("E", "K", "N", "num_tokens", "topk", "in_dtype")):
            continue
        if key["in_dtype"] != in_dtype:
            continue
        candidates.append(key)

    if not candidates:
        _pick_cache[cache_key] = None
        return None

    def sort_key(key: CaseKey) -> tuple[int, int, int, int, int, int]:
        nt = key["num_tokens"]
        covers = 0 if nt >= num_tokens else 1
        return (
            abs(key["topk"] - top_k),
            abs(key["E"] - E),
            abs(key["K"] - K),
            abs(key["N"] - N),
            covers,
            abs(nt - num_tokens),
        )

    result = min(candidates, key=sort_key)
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    output: torch.Tensor,  # [num_tokens, H] (reduced layer output)
    hidden_states: torch.Tensor,  # [num_tokens, K] fp8
    w1: torch.Tensor,  # [E, N, K] fp8 (gate_up, N = 2 * intermediate)
    w2: torch.Tensor,  # [E, H, intermediate] fp8 (down projection)
    w1_scale: torch.Tensor,  # [E, N // 128, K // 128]
    w2_scale: torch.Tensor,  # [E, H // 128, intermediate // 128]
    a1q_scale: torch.Tensor,  # [num_tokens, K // 128]
    topk_weights: torch.Tensor,  # [num_tokens, top_k] routed combine weights
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    expert_map: torch.Tensor | None,  # [global_num_experts] or None
    global_num_experts: int,
    workspace13: torch.Tensor,  # scratch for intermediate scales
    workspace2: torch.Tensor,  # scratch for quantized intermediate
) -> None:
    return


def baseline(
    output: torch.Tensor,  # [num_tokens, H] out_dtype (reduced layer output)
    hidden_states: torch.Tensor,  # [num_tokens, K] fp8
    w1: torch.Tensor,  # [E, N, K] fp8 (gate_up, N = 2 * intermediate)
    w2: torch.Tensor,  # [E, H, intermediate] fp8 (down projection)
    w1_scale: torch.Tensor,  # [E, N // 128, K // 128]
    w2_scale: torch.Tensor,  # [E, H // 128, intermediate // 128]
    a1q_scale: torch.Tensor,  # [num_tokens, K // 128]
    topk_weights: torch.Tensor,  # [num_tokens, top_k] routed combine weights
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    expert_map: torch.Tensor | None,  # [global_num_experts] or None
    global_num_experts: int,
    workspace13: torch.Tensor,  # scratch, unused by this reference
    workspace2: torch.Tensor,  # scratch, unused by this reference
) -> None:
    # Full MoE: GEMM1 (gate_up) + silu_and_mul + per-block fp8 quant of the
    # intermediate + GEMM2 (down projection) with the routed weight applied
    # per row, then reduced over top_k into the layer output. Routed row r is
    # token r // top_k routed to expert topk_ids.flatten()[r], matching the
    # kernel's sorted_token_ids scatter. gate_up is rounded through the output
    # dtype to mirror the kernel's bf16 gate_up handoff between its GEMM1 and
    # silu+quant tile loops.
    #
    # The workspace caches only stage the kernel's gate_up / fp8 intermediate
    # and are excluded from the autotuner accuracy check (see
    # _baseline_accuracy_check), so the reference leaves them untouched.
    num_tokens, top_k = topk_ids.shape
    routed = num_tokens * top_k
    N = w1.shape[1]
    H = w2.shape[1]
    intermediate = N // 2
    qtype_max = torch.finfo(torch.float8_e4m3fn).max
    min_scaling_factor = 1.0 / (qtype_max * 512.0)
    a_full = hidden_states.to(torch.float32) * a1q_scale.to(
        torch.float32
    ).repeat_interleave(GROUP_K, dim=1)
    w_flat = topk_weights.reshape(-1).to(torch.float32)
    out_rows = torch.zeros((routed, H), dtype=output.dtype, device=output.device)
    flat_e = topk_ids.reshape(-1).tolist()
    for r, e in enumerate(flat_e):
        if expert_map is not None:
            e = int(expert_map[e])
        if e < 0:
            continue
        w1_e = w1[e].to(torch.float32)
        w1_s = (
            w1_scale[e]
            .to(torch.float32)
            .repeat_interleave(GROUP_N, dim=0)
            .repeat_interleave(GROUP_K, dim=1)
        )
        gate_up = (
            (a_full[r // top_k] @ (w1_e * w1_s).t()).to(output.dtype).to(torch.float32)
        )
        gate = gate_up[:intermediate]
        up = gate_up[intermediate:]
        h = (gate * torch.sigmoid(gate) * up).view(-1, GROUP_N)
        s = (h.abs().amax(dim=-1) / qtype_max).clamp(min=min_scaling_factor)
        hq = (h / s[:, None]).clamp(-qtype_max, qtype_max).to(torch.float8_e4m3fn)

        # Dequantize the fp8 intermediate, then the weighted GEMM2 (down
        # projection).
        h_deq = (hq.to(torch.float32) * s[:, None]).reshape(intermediate)
        w2_e = w2[e].to(torch.float32)
        w2_s = (
            w2_scale[e]
            .to(torch.float32)
            .repeat_interleave(GROUP_N, dim=0)
            .repeat_interleave(GROUP_N, dim=1)
        )
        result = h_deq @ (w2_e * w2_s).t()
        out_rows[r] = (result * w_flat[r]).to(output.dtype)
    output.copy_(out_rows.view(num_tokens, top_k, H).sum(dim=1).to(output.dtype))


# Index of ``output`` in moe_block_mm's signature. ``output`` is the only mutated
# arg worth checking; the other mutated args (workspace13 at 10, workspace2 at 11)
# only stage the gate_up / fp8 intermediate scratch, which the reference does not
# reproduce, so they are excluded from the autotuner's post-run compare.
_OUTPUT_ARG_INDEX = 0


def _baseline_accuracy_check(actual: object, expected: object) -> None:
    """Autotuner accuracy check that ignores the scratch workspace caches.

    Helion invokes this both for the kernel return value (None here) and for the
    post-run argument tuple; for the latter, compare only ``output`` and skip the
    workspace caches, which hold fp8 quant scratch the reference does not fill.
    """
    if isinstance(actual, tuple | list):
        assert isinstance(expected, tuple | list)
        assert_close_with_mismatch_tolerance(
            actual[_OUTPUT_ARG_INDEX],
            expected[_OUTPUT_ARG_INDEX],
            atol=1e-1,
            rtol=1e-1,
        )
    elif actual is not None:
        assert_close_with_mismatch_tolerance(actual, expected, atol=1e-1, rtol=1e-1)


def _align_sizes(
    num_tokens: int, top_k: int, E: int, m_block_size: int
) -> tuple[int, int]:
    numel = num_tokens * top_k
    max_padded = numel + E * (m_block_size - 1)
    if numel < E:
        max_padded = min(numel * m_block_size, max_padded)
    num_blocks = (max_padded + m_block_size - 1) // m_block_size
    return max_padded, num_blocks


@torch.library.custom_op("vllm_helion::moe_block_align", mutates_args=())
def _moe_block_align_op(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    m_block_size: int,
    global_num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Dense layout at block_size 1 (and no expert_map): block b is routed
    # slot b, so the GEMM loops derive the slot mapping from the block id
    # (Triton fused_moe's naive_block_assignment) and never load the sorted
    # ids. The returned views cost no device work, and their shapes match the
    # fake impl's formulas at m_block_size == 1 (max_padded == num_valid).
    if m_block_size == 1 and expert_map is None:
        flat = topk_ids.reshape(-1)
        return flat, flat, flat[:1]
    return moe_align_block_size(topk_ids, m_block_size, global_num_experts, expert_map)


@_moe_block_align_op.register_fake
def _(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    m_block_size: int,
    global_num_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_padded, num_blocks = _align_sizes(
        topk_ids.shape[0], topk_ids.shape[1], global_num_experts, m_block_size
    )
    sorted_ids = topk_ids.new_empty((max_padded,), dtype=torch.int32)
    expert_ids = topk_ids.new_empty((num_blocks,), dtype=torch.int32)
    num_tokens_post_pad = topk_ids.new_empty((1,), dtype=torch.int32)
    return sorted_ids, expert_ids, num_tokens_post_pad


# Quantized GEMM kernels can have relatively large numerical differences from
# the baseline, so the accuracy check runs at a loose 1e-1 tolerance. It also
# compares only ``output``: the workspace13 / workspace2 caches hold gate_up /
# fp8 quant scratch the reference does not reproduce, so
# autotune_baseline_accuracy_check_fn excludes them (see
# _baseline_accuracy_check) to prevent spurious config failures.
@register_kernel(
    mutates_args=["output", "workspace13", "workspace2"],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    use_variant_config=True,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
        autotune_baseline_accuracy_check_fn=_baseline_accuracy_check,
        ignore_warnings=[
            helion.exc.TensorOperationInWrapper,
            helion.exc.ProcessGroupNameNotFound,
        ],
    ),
)
def fused_moe_block_mm(
    output: torch.Tensor,  # [num_tokens, H] (reduced layer output)
    a1q: torch.Tensor,  # [num_tokens, H]
    w1: torch.Tensor,  # [E, 2I, H]
    w2: torch.Tensor,  # [E, H, I]
    w1_scale: torch.Tensor,  # [E, 2I // 128, H // 128]
    w2_scale: torch.Tensor,  # [E, H // 128, I // 128]
    a1q_scale: torch.Tensor,  # [num_tokens, H // 128]
    topk_weights: torch.Tensor,  # [num_tokens, top_k] routed combine weights
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    expert_map: torch.Tensor | None,  # [global_num_experts] or None
    global_num_experts: int,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
) -> None:
    num_tokens, H = a1q.shape
    top_k = topk_ids.shape[1]
    E = w1.shape[0]
    H = w2.shape[1]
    intermediate = w2.shape[2]
    N = w1.shape[1]
    assert w1.shape[1] == 2 * intermediate

    hl.specialize(num_tokens)
    hl.specialize(H)
    hl.specialize(E)
    hl.specialize(intermediate)
    hl.specialize(top_k)
    # N is a GEMM1 grid extent; specializing it keeps the loop's task count
    # concrete so cross_loop_pipeline='static'/'dynamic' can size the persistent
    # task universe (a symbolic count forces the barrier fallback).
    hl.specialize(N)

    assert w1.shape[-1] == H
    assert w2.shape == (E, H, intermediate)
    assert a1q.dtype == w1.dtype == w2.dtype
    assert a1q.stride(1) == 1
    assert w1.stride(2) == 1
    assert w2.stride(2) == 1
    assert topk_ids.shape[0] == num_tokens
    assert output.shape == (num_tokens, H)

    # Routed combine weights (the MoE top-k coefficients), one value per
    # routed slot, applied per GEMM2 output row in the epilogue (Triton's
    # mul_routed_weight). reshape(-1) always yields a stride-1 flat tensor
    # (view or copy).
    topk_weights_flat = topk_weights.reshape(-1)
    assert topk_weights_flat.numel() == num_tokens * top_k
    assert topk_weights_flat.dtype.is_floating_point

    out_dtype = output.dtype
    assert out_dtype.is_floating_point

    assert a1q_scale.ndim == 2 and w1_scale.ndim == 3 and w2_scale.ndim == 3
    assert a1q_scale.dtype == w1_scale.dtype and a1q_scale.is_floating_point()
    assert w2_scale.dtype == w1_scale.dtype

    num_group_m, num_group_h = a1q_scale.shape
    num_group_i = w2_scale.shape[-1]
    assert w1_scale.shape == (E, num_group_i * 2, num_group_h)
    assert w2_scale.shape[0] == E
    assert w2_scale.shape[1] == num_group_h

    assert num_group_m == num_tokens
    assert H % num_group_h == 0
    assert intermediate % num_group_i == 0

    # a1q_scale group shape must be [1, 128], w1_scale group shape [128, 128].
    group_i = intermediate // num_group_i
    group_h = H // num_group_h
    assert group_i == 128 and group_h == 128

    hl.specialize(num_group_h)
    hl.specialize(num_group_i)

    hl.specialize(group_h)
    hl.specialize(group_i)

    # w2_scale: [E, num_group_h, num_group_ni], 128x128 blocks over (H, I).
    assert w2_scale.shape == (E, num_group_h, num_group_i)

    acc_dtype = torch.float32 if a1q.is_floating_point() else torch.int32

    # Per-block fp8 (e4m3) quant constants for the intermediate, matching
    # silu_and_mul_per_block_quant: clamp scales to a floor to bound dequant.
    # The intermediate is e4m3 regardless of out_dtype (torch.finfo is not
    # traceable inside the Helion kernel body).
    qtype_max = 448.0
    qtype_min = -qtype_max
    min_scaling_factor = 1.0 / (qtype_max * 512.0)

    # m_block_size is the moe_align block size: each expert block holds up to
    # this many routed rows. Under num_tokens*top_k < E the padded space is
    # capped at numel*m_block_size, so there are at most num_valid blocks and
    # expert_ids has exactly num_valid entries. The block loops therefore tile
    # the block space [0, num_valid) one block at a time. At m_block_size == 1
    # (and no expert_map) the align op returns the dense layout instead --
    # block b is routed slot b, derived in-kernel (Triton fused_moe's
    # naive_block_assignment) -- so no alignment kernels run.
    m_block_size = hl.register_block_size(num_tokens)
    sorted_token_ids, expert_ids, _ = torch.ops.vllm_helion.moe_block_align(
        topk_ids, expert_map, m_block_size, global_num_experts
    )
    max_padded = sorted_token_ids.size(0)
    num_valid = num_tokens * top_k
    hl.specialize(num_valid)

    # Intermediates staged in the preallocated workspace caches (metadata-only
    # resize, no allocation). gate_up is the bf16 [num_valid, N] projection in
    # routed-row order, resized from workspace2 like the Triton baseline
    # (TritonExperts' intermediate_cache1). workspace13 holds the fp8 [num_valid,
    # I] intermediate from its base and the f32 [num_valid, num_group_i] scales
    # just past it (byte offset num_valid * I, 4-aligned since I % 128 == 0; the
    # remaining 2*max(I, H) - I >= I bytes per routed row always cover both).
    gate_up = _resize_cache(workspace2, (num_tokens, top_k, N)).view(num_valid, N)
    a2q = _resize_cache(workspace13.view(dtype=a1q.dtype), (num_valid, intermediate))
    ws13_f32 = workspace13.view(dtype=a1q_scale.dtype).flatten()
    scale_off = (num_valid * intermediate) // 4
    a2q_scale = ws13_f32[scale_off : scale_off + num_valid * num_group_i].view(
        num_valid, num_group_i
    )
    gate_up_3d = gate_up.view(num_valid, -1, group_i)
    a2q_3d = a2q.view(num_valid, -1, group_i)

    # GEMM2's per-slot output, staged in workspace2 past gate_up so the two
    # never overlap (a launch-time zero of mm2_out must survive GEMM1's
    # gate_up writes); the allocation is [num_tokens, top_k, N + H].
    assert workspace2.numel() >= num_valid * (N + H)
    mm2_3d = workspace2.flatten()[num_valid * N :].view(num_tokens, top_k, H)
    mm2_out = mm2_3d.view(num_valid, H)

    # GEMM2's per-slot output for the split_k_2 == 1 path, staged in
    # workspace2's gate_up region: gate_up is dead once the silu+quant loop
    # consumed it, and [num_tokens, top_k, H] always fits the [num_tokens,
    # top_k, max(N, H)] allocation. Derived before the first tile loop:
    # helion forbids statements between top-level loops.
    mm2_3d = _resize_cache(workspace2, (num_tokens, top_k, H))
    mm2_out = mm2_3d.view(num_valid, H)

    swap_ab_1 = hl.register_tunable("swap_ab_1", BooleanFragment())
    swap_ab_2 = hl.register_tunable("swap_ab_2", BooleanFragment())

    # Split-K over each GEMM's reduction dim: H (hidden) for GEMM1, I
    # (intermediate) for GEMM2.
    split_k_1 = hl.register_tunable(
        "split_k_1", PowerOfTwoFragment(1, helion.next_power_of_2(num_group_h))
    )
    k_block_size_1 = helion.next_power_of_2(helion.cdiv(H, split_k_1))
    if split_k_1 > 1:
        gate_up.zero_()

    split_k_2 = hl.register_tunable(
        "split_k_2", PowerOfTwoFragment(1, helion.next_power_of_2(num_group_i))
    )
    k_block_size_2 = helion.next_power_of_2(helion.cdiv(intermediate, split_k_2))

    if split_k_2 > 1:
        mm2_out.zero_()

    # GEMM1: gate_up projection over the full N width (free N tiling now that
    # silu+quant is a separate loop), gathering the per-token activation by
    # offs_token // top_k.
    for tile_gm1, tile_n1, outer_k1 in hl.tile(
        [num_valid, N, H], block_size=[1, None, k_block_size_1]
    ):
        blk1 = tile_gm1.begin
        e1 = expert_ids[blk1]

        # Skip fully-padded blocks: moe_align fills the expert_ids tail with -1
        # (and expert_map marks off-rank experts -1), so this is Triton's
        # `off_experts == -1` early return. `e1` is program-uniform, so the
        # branch drops the whole MMA for no-op tiles without lane divergence.
        if e1 >= 0:
            # Expand this block to its m_block_size sorted rows. The block may
            # spill past sorted_token_ids (padding tail), so mask spill lanes.
            rows1 = blk1 * m_block_size + hl.arange(m_block_size)
            if m_block_size == 1 and expert_map is None:
                # Dense layout: block b is routed slot b, derived
                # arithmetically -- no sorted-ids load, no padding.
                offs_token1 = rows1
                token_mask1 = rows1 < num_valid
            else:
                row_mask1 = rows1 < max_padded
                offs_token1 = hl.load(sorted_token_ids, [rows1], extra_mask=row_mask1)
                token_mask1 = row_mask1 & (offs_token1 < num_valid)
            a_rows1 = torch.where(token_mask1, offs_token1 // top_k, 0)

            acc1 = hl.zeros([m_block_size, tile_n1], torch.float32)

            # Keep block_size = group_h for the H dimension so the block scale
            # is constant across each inner tile.
            for tile_h1 in hl.tile(outer_k1.begin, outer_k1.end, block_size=group_h):
                a_blk1 = hl.load(
                    a1q, [a_rows1, tile_h1], extra_mask=token_mask1[:, None]
                )
                w1_blk = w1[e1, tile_n1, tile_h1]

                if swap_ab_1:
                    acc_blk1 = (
                        hl.dot(w1_blk, a_blk1.t(), out_dtype=acc_dtype)
                        .t()
                        .to(torch.float32)
                    )
                else:
                    acc_blk1 = hl.dot(a_blk1, w1_blk.t(), out_dtype=acc_dtype).to(
                        torch.float32
                    )

                gh_idx1 = tile_h1.id
                a_scales_blk1 = hl.load(
                    a1q_scale,
                    [a_rows1, gh_idx1 + hl.arange(1)],
                    extra_mask=token_mask1[:, None],
                )
                w1_scale_blk = w1_scale[e1, tile_n1.index // group_i, gh_idx1][None, :]
                acc1 = acc1 + (a_scales_blk1 * w1_scale_blk) * acc_blk1

            out_rows1 = torch.where(token_mask1, offs_token1, 0)

            if split_k_1 == 1:
                hl.store(
                    gate_up,
                    [out_rows1, tile_n1],
                    acc1.to(out_dtype),
                    extra_mask=token_mask1[:, None],
                )
            else:
                # Masked lanes contribute 0 (their a1q / scale loads are masked
                # to 0), so redirect their atomic to a valid row.
                hl.atomic_add(gate_up, [out_rows1, tile_n1], acc1.to(out_dtype))

    for tile_sm, tile_sg, tile_sn in hl.tile(
        [num_valid, num_group_i, group_i], block_size=[1, None, group_i]
    ):
        x_a_blk = gate_up_3d[tile_sm, tile_sg, tile_sn].to(torch.float32)
        x_b_blk = hl.load(
            gate_up_3d,
            [tile_sm, tile_sg.index + num_group_i, tile_sn],
            extra_mask=(tile_sg.index + num_group_i < 2 * num_group_i)[None, :, None],
        ).to(torch.float32)
        x_blk = x_a_blk * torch.sigmoid(x_a_blk) * x_b_blk
        s_blk = torch.amax(torch.abs(x_blk), dim=-1).to(torch.float32)
        s_blk = (s_blk * (1.0 / qtype_max)).clamp(min=min_scaling_factor)
        a2q_scale[tile_sm, tile_sg] = s_blk
        y_blk = x_blk / s_blk[:, :, None]
        a2q_3d[tile_sm, tile_sg, tile_sn] = y_blk.clamp(qtype_min, qtype_max).to(
            a2q.dtype
        )

    # GEMM2: down projection of the quantized intermediate (dense routed-row
    # order) back to the model hidden dim, each row scaled by its routed
    # weight (Triton's mul_routed_weight), staged per routed row in mm2_out --
    # stored whole at split_k_2 == 1, atomically accumulated over the split-K
    # slices otherwise (mirroring GEMM1's split_k_1 pattern).
    for tile_gm2, tile_h2, outer_k2 in hl.tile(
        [num_valid, H, intermediate], block_size=[1, None, k_block_size_2]
    ):
        blk2 = tile_gm2.begin
        e2 = expert_ids[blk2]

        if e2 >= 0:
            rows2 = blk2 * m_block_size + hl.arange(m_block_size)
            if m_block_size == 1 and expert_map is None:
                # Dense layout: block b is routed slot b, derived
                # arithmetically -- no sorted-ids load, no padding.
                offs_token2 = rows2
                token_mask2 = rows2 < num_valid
            else:
                row_mask2 = rows2 < max_padded
                offs_token2 = hl.load(sorted_token_ids, [rows2], extra_mask=row_mask2)
                token_mask2 = row_mask2 & (offs_token2 < num_valid)
            # a2q is in routed-row order, so gather by offs_token directly.
            a2q_rows = torch.where(token_mask2, offs_token2, 0)

            acc = hl.zeros([m_block_size, tile_h2], torch.float32)

            for tile_i2 in hl.tile(outer_k2.begin, outer_k2.end, block_size=group_i):
                a2q_blk = hl.load(
                    a2q, [a2q_rows, tile_i2], extra_mask=token_mask2[:, None]
                )
                w2_blk = w2[e2, tile_h2, tile_i2]

                if swap_ab_2:
                    acc_blk = (
                        hl.dot(w2_blk, a2q_blk.t(), out_dtype=acc_dtype)
                        .t()
                        .to(torch.float32)
                    )
                else:
                    acc_blk = hl.dot(a2q_blk, w2_blk.t(), out_dtype=acc_dtype).to(
                        torch.float32
                    )

                gi_idx = tile_i2.id
                a2q_scale_blk = hl.load(
                    a2q_scale,
                    [a2q_rows, gi_idx + hl.arange(1)],
                    extra_mask=token_mask2[:, None],
                )
                w2_scale_blk = w2_scale[e2, tile_h2.index // group_h, gi_idx][None, :]
                acc = acc + (a2q_scale_blk * w2_scale_blk) * acc_blk

            out_rows = torch.where(token_mask2, offs_token2, 0)

            # Routed weight per row (Triton's mul_routed_weight). Split-K
            # partials are disjoint K-slices of the same output row, so
            # weighting each partial equals weighting the total: sum_k(w * P_k)
            # == w * P.
            weight_blk = hl.load(topk_weights_flat, [out_rows], extra_mask=token_mask2)
            acc = acc * weight_blk[:, None]

            if split_k_2 == 1:
                # No split-K partials: store the per-slot row for the moe_sum
                # loop below (each routed row written exactly once).
                hl.store(
                    mm2_out,
                    [out_rows, tile_h2],
                    acc.to(out_dtype),
                    extra_mask=token_mask2[:, None],
                )
            else:
                # Masked lanes contribute 0 (their a2q / scale loads are masked
                # to 0), so redirect their atomic to a valid row.
                hl.atomic_add(mm2_out, [out_rows, tile_h2], acc.to(out_dtype))

    # Combine: reduce the weighted routed rows over top_k into the layer
    # output (ops.moe_sum's reduction). GEMM2 wrote every routed row (stored
    # whole at split_k_2 == 1, accumulated over the split-K slices otherwise),
    # so no mask is needed.
    for tile_m, tile_h in hl.tile([num_tokens, H], block_size=[None, None]):
        acc_sum = hl.zeros([tile_m, tile_h], torch.float32)
        for k in range(top_k):
            acc_sum = acc_sum + mm2_3d[tile_m, k, tile_h].to(torch.float32)
        output[tile_m, tile_h] = acc_sum.to(out_dtype)
