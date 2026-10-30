# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand-written custom baselines for benchmarking Helion kernels.

Each entry in ``CUSTOM_BASELINE_FNS`` maps a Helion kernel name to a reference
callable sharing that kernel's argument interface, so the kernel's input tuple
can be forwarded verbatim. These are used as the ``custom`` performance baseline
(e.g. by ``scripts/benchmark_helion_kernels.py``) to compare a Helion kernel
against a production vLLM op rather than the native-torch autotune reference.

Add a new baseline by defining a function with the kernel's argument interface
and registering it in ``CUSTOM_BASELINE_FNS``.
"""

import weakref
from collections.abc import Callable
from typing import NamedTuple

import torch


def _block_scaled_mm(
    out: torch.Tensor,  # [M, N]
    a: torch.Tensor,  # [M, K]
    b: torch.Tensor,  # [K, N]
    a_scales: torch.Tensor,  # [num_group_m, num_group_k]
    b_scales: torch.Tensor,  # [num_group_k, num_group_n]
) -> None:
    from vllm.model_executor.kernels.linear.scaled_mm.deep_gemm import fp8_gemm_nt

    fp8_gemm_nt((a, a_scales), (b.T, b_scales.T), out, is_deep_gemm_e8m0_used=True)


def _moe_block_mm(
    output: torch.Tensor,  # [num_tokens, H] bf16 (reduced layer output)
    hidden_states: torch.Tensor,  # [num_tokens, K] fp8
    w1: torch.Tensor,  # [E, N, K] fp8 (gate_up, N = 2 * intermediate)
    w2: torch.Tensor,  # [E, H, intermediate] fp8 (down projection)
    w1_scale: torch.Tensor,  # [E, N // 128, K // 128] f32
    w2_scale: torch.Tensor,  # [E, H // 128, intermediate // 128] f32
    a1q_scale: torch.Tensor,  # [num_tokens, K // 128] f32
    topk_weights: torch.Tensor,  # [num_tokens, top_k] f32
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    expert_map: torch.Tensor | None,  # [global_num_experts] or None
    global_num_experts: int,
    workspace13: torch.Tensor,  # bf16 [num_tokens, top_k, H] GEMM2 output cache
    workspace2: torch.Tensor,  # bf16 [num_tokens, top_k, N] gate_up cache
) -> None:
    """Triton GEMM1 + silu block-quant + weighted GEMM2 + moe_sum baseline for
    ``fused_moe_block_mm``.

    Mirrors ``TritonExperts.apply``: a single ``_prepare_expert_assignment`` (and
    a single ``try_get_optimal_moe_config``) feeds both GEMMs. GEMM1 drives
    ``invoke_fused_moe_triton_kernel`` over the unpermuted activations to produce
    the bf16 gate_up projection in routed-row (``offs_token``) order;
    ``silu_and_mul_per_block_quant`` folds the gate||up halves into the fp8
    [routed, intermediate] intermediate + per-128-group scales; GEMM2 reuses the
    same sorted_token_ids/expert_ids with ``top_k = 1`` (so ``offs_token``
    indexes the intermediate directly) and applies the routed weights in its
    epilogue (``mul_routed_weight``) into workspace13; ``moe_sum`` reduces the
    weighted rows over top_k into the [num_tokens, H] output. Runs on any
    Triton-capable GPU (e.g. SM120).
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _prepare_expert_assignment,
        invoke_fused_moe_triton_kernel,
        try_get_optimal_moe_config,
    )
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache
    from vllm.triton_utils import tl

    E, N, K = w1.shape
    H = w2.shape[1]
    num_tokens, top_k = topk_ids.shape
    routed = num_tokens * top_k

    if hidden_states.dtype != torch.float8_e4m3fn:
        raise NotImplementedError(
            "The Triton fused_moe block-scaled baseline only supports fp8 "
            f"activations, got {hidden_states.dtype}."
        )

    # moe_block_mm quantizes with 128x128 weight blocks and 1x128 act groups.
    block_shape = [128, 128]
    compute_type = tl.bfloat16

    # One config + one assignment shared by both GEMMs (TritonExperts.apply).
    # _prepare_expert_assignment skips moe_align_block_size entirely in the sparse
    # low-latency regime (expert_map is None and num_tokens*top_k*4 <= experts),
    # returning sorted_ids=None to drive the kernel's naive_block_assignment path.
    config = try_get_optimal_moe_config(
        w1.shape, w2.shape, top_k, "fp8_w8a8", num_tokens, block_shape=block_shape
    )
    num_align_experts = E if expert_map is None else global_num_experts
    sorted_ids, expert_ids, ntpp = _prepare_expert_assignment(
        topk_ids,
        config,
        num_tokens,
        top_k,
        num_align_experts,
        expert_map,
        block_shape=block_shape,
    )

    # GEMM1: gate_up projection to bf16 [num_tokens, top_k, N], written in routed
    # (offs_token) order. Reuse the preallocated bf16 workspace for the gate_up
    # cache (like TritonExperts.apply's intermediate_cache1) so the timed region
    # allocates nothing; the fused silu+quant below mints its own fp8 output,
    # matching the TritonExperts fp8 fast path which leaves workspace13 unused.
    gate_up = _resize_cache(workspace2, (num_tokens, top_k, N))
    invoke_fused_moe_triton_kernel(
        hidden_states,
        w1,
        gate_up,
        a1q_scale,
        w1_scale,
        None,
        sorted_ids,
        expert_ids,
        ntpp,
        False,
        top_k,
        config,
        compute_type,
        True,  # use_fp8_w8a8
        False,  # use_int8_w8a8
        False,  # use_int8_w8a16
        False,  # use_int4_w4a16
        False,  # per_channel_quant
        block_shape,
    )

    # silu_and_mul + per-block fp8 quant of the intermediate.
    hq, hq_scale = ops.silu_and_mul_per_block_quant(
        gate_up.view(routed, N),
        group_size=128,
        quant_dtype=torch.float8_e4m3fn,
    )

    # GEMM2: down projection reusing the GEMM1 alignment. top_k = 1 makes the
    # kernel's offs_token // top_k index the routed-row intermediate directly, so
    # the shared sorted_token_ids/expert_ids apply unchanged. The routed
    # weights are applied in the epilogue (mul_routed_weight) into
    # workspace13, then reduced over top_k.
    mm2 = _resize_cache(workspace13, (num_tokens, top_k, H))
    invoke_fused_moe_triton_kernel(
        hq,
        w2,
        mm2.view(routed, 1, H),
        hq_scale,
        w2_scale,
        topk_weights,
        sorted_ids,
        expert_ids,
        ntpp,
        True,  # mul_routed_weight
        1,
        config,
        compute_type,
        True,  # use_fp8_w8a8
        False,  # use_int8_w8a8
        False,  # use_int8_w8a16
        False,  # use_int4_w4a16
        False,  # per_channel_quant
        block_shape,
    )
    ops.moe_sum(mm2, output)


def triton_moe_block_mm(
    output: torch.Tensor,  # [num_tokens, H] bf16 (reduced layer output)
    hidden_states: torch.Tensor,  # [num_tokens, K] fp8
    w1: torch.Tensor,  # [E, N, K] fp8 (gate_up, N = 2 * intermediate)
    w2: torch.Tensor,  # [E, H, intermediate] fp8 (down projection)
    w1_scale: torch.Tensor,  # [E, N // 128, K // 128] f32
    w2_scale: torch.Tensor,  # [E, H // 128, intermediate // 128] f32
    a1q_scale: torch.Tensor,  # [num_tokens, K // 128] f32
    topk_weights: torch.Tensor,  # [num_tokens, top_k] f32
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    expert_map: torch.Tensor | None,  # [global_num_experts] or None
    global_num_experts: int,
    workspace13: torch.Tensor,  # bf16 [routed, H] GEMM2 output cache
    workspace2: torch.Tensor,  # bf16 [routed, N] gate_up cache
    clamp_limit: float | None = None,  # swiglu_limit of the gated activation
) -> None:
    """Triton full-layer baseline for the composite ``mega_moe_block_mm``.

    Mirrors ``TritonExperts.apply`` end to end: one config and one
    ``_prepare_expert_assignment`` feed both GEMMs; GEMM1 writes the bf16
    gate_up projection in routed-row order; ``silu_and_mul_per_block_quant``
    produces the fp8 intermediate; GEMM2 applies the routed weights in its
    epilogue (``mul_routed_weight``) into workspace13; ``moe_sum`` reduces the
    weighted rows over top_k into the [num_tokens, H] output. The gate_up and
    GEMM2 staging buffers are workspace-backed so the timed region allocates
    nothing. Runs on any Triton-capable GPU (e.g. SM120).
    """
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        _prepare_expert_assignment,
        invoke_fused_moe_triton_kernel,
        try_get_optimal_moe_config,
    )
    from vllm.model_executor.layers.fused_moe.utils import (
        _resize_cache,
        moe_kernel_quantize_input,
        swiglu_limit_func,
    )
    from vllm.triton_utils import tl

    E, N, K = w1.shape
    num_tokens, top_k = topk_ids.shape
    routed = num_tokens * top_k
    H = w2.shape[1]

    if hidden_states.dtype != torch.float8_e4m3fn:
        raise NotImplementedError(
            "The Triton fused_moe block-scaled baseline only supports fp8 "
            f"activations, got {hidden_states.dtype}."
        )

    # moe_block_mm quantizes with 128x128 weight blocks and 1x128 act groups.
    block_shape = [128, 128]
    compute_type = tl.bfloat16

    # One config + one assignment, computed exactly as TritonExperts.apply
    # does for the full MoE.
    config = try_get_optimal_moe_config(
        w1.shape, w2.shape, top_k, "fp8_w8a8", num_tokens, block_shape=block_shape
    )
    num_align_experts = E if expert_map is None else global_num_experts
    sorted_ids, expert_ids, ntpp = _prepare_expert_assignment(
        topk_ids,
        config,
        num_tokens,
        top_k,
        num_align_experts,
        expert_map,
        block_shape=block_shape,
    )

    # GEMM1: gate_up projection staged in workspace2 (TritonExperts'
    # intermediate_cache1 role), in routed (offs_token) order.
    gate_up = _resize_cache(workspace2, (num_tokens, top_k, N))
    invoke_fused_moe_triton_kernel(
        hidden_states,
        w1,
        gate_up,
        a1q_scale,
        w1_scale,
        None,
        sorted_ids,
        expert_ids,
        ntpp,
        False,
        top_k,
        config,
        compute_type,
        True,  # use_fp8_w8a8
        False,  # use_int8_w8a8
        False,  # use_int8_w8a16
        False,  # use_int4_w4a16
        False,  # per_channel_quant
        block_shape,
    )

    # silu_and_mul + per-block fp8 quant of the intermediate. A clamp limit
    # takes the two steps TritonExperts falls back to (swiglu_limit_func, then
    # the block quantization) since the fused op has no clamp argument.
    if clamp_limit is None:
        hq, hq_scale = ops.silu_and_mul_per_block_quant(
            gate_up.view(routed, N),
            group_size=128,
            quant_dtype=torch.float8_e4m3fn,
        )
    else:
        act = torch.empty((routed, N // 2), dtype=gate_up.dtype, device=gate_up.device)
        swiglu_limit_func(act, gate_up.view(routed, N), clamp_limit)
        hq, hq_scale = moe_kernel_quantize_input(
            act, None, torch.float8_e4m3fn, False, [128, 128]
        )

    # GEMM2: down projection with the routed weights applied in the epilogue
    # (mul_routed_weight), staged in workspace13 (TritonExperts'
    # intermediate_cache3 role), then reduced over top_k.
    mm2 = _resize_cache(workspace13, (num_tokens, top_k, H))
    invoke_fused_moe_triton_kernel(
        hq,
        w2,
        mm2.view(routed, 1, H),
        hq_scale,
        w2_scale,
        topk_weights,
        sorted_ids,
        expert_ids,
        ntpp,
        True,  # mul_routed_weight
        1,
        config,
        compute_type,
        True,  # use_fp8_w8a8
        False,  # use_int8_w8a8
        False,  # use_int8_w8a16
        False,  # use_int4_w4a16
        False,  # per_channel_quant
        block_shape,
    )
    ops.moe_sum(mm2, output)


# vLLM's CUTLASS block-scale path floors near-zero block scales to avoid NaNs
# in the kernel (see prepare_fp8_moe_layer_for_fi).
_FI_MIN_BLOCK_SCALE = 1e-10


class _FiState(NamedTuple):
    """Everything ``FlashInferExperts`` builds before the per-call work.

    Mirrors the split in ``FlashInferExperts``: the weight layout and scale
    floor come from load time (``convert_to_kernel_format`` ->
    ``prepare_fp8_moe_layer_for_fi``), while ``ep_size``/``ep_rank`` and the
    per-expert ``gemm1_clamp_limit`` tensor are ``__init__`` attributes. None of
    it is recomputed per request, so the baseline caches it the same way.
    """

    w13: torch.Tensor
    w13_scale: torch.Tensor
    fc2_scale: torch.Tensor
    swiglu_limit: torch.Tensor | None
    ep_size: int
    ep_rank: int


# id(w1) -> (weakref to w1, key, state). The weakref callback drops the entry
# when the source weight dies, so the converted (large) tensors are not held
# beyond the case that produced them.
_FI_CACHE: dict[int, tuple[weakref.ref, tuple[object, ...], _FiState]] = {}


def _fi_state(
    w1: torch.Tensor,
    w2: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    expert_map: torch.Tensor | None,
    global_num_experts: int,
    clamp_limit: float | None,
) -> _FiState:
    """Load-time and init-time state of ``FlashInferExperts``."""
    key = (
        id(w2),
        id(w1_scale),
        id(w2_scale),
        id(expert_map),
        global_num_experts,
        clamp_limit,
    )
    cache_id = id(w1)
    entry = _FI_CACHE.get(cache_id)
    if entry is not None and entry[1] == key and entry[0]() is w1:
        return entry[2]

    def swap_gate_up(x: torch.Tensor) -> torch.Tensor:
        # vLLM stores [gate; up]; FI's CUTLASS kernels want W31.
        return (
            x.reshape(-1, 2, x.shape[-2] // 2, x.shape[-1])
            .flip(dims=[1])
            .reshape(x.shape)
        )

    local_num_experts = w1.shape[0]
    ep_size, ep_rank = _fi_ep_params(expert_map, local_num_experts, global_num_experts)
    state = _FiState(
        w13=swap_gate_up(w1),
        w13_scale=swap_gate_up(w1_scale).clamp(min=_FI_MIN_BLOCK_SCALE),
        fc2_scale=w2_scale.clamp(min=_FI_MIN_BLOCK_SCALE),
        swiglu_limit=(
            None
            if clamp_limit is None
            else torch.full(
                (local_num_experts,),
                float(clamp_limit),
                dtype=torch.float32,
                device=w1.device,
            )
        ),
        ep_size=ep_size,
        ep_rank=ep_rank,
    )
    _FI_CACHE[cache_id] = (
        weakref.ref(w1, lambda _, entry_id=cache_id: _FI_CACHE.pop(entry_id, None)),
        key,
        state,
    )
    return state


def _fi_ep_params(
    expert_map: torch.Tensor | None,
    local_num_experts: int,
    global_num_experts: int,
) -> tuple[int, int]:
    """``(ep_size, ep_rank)`` telling FI which expert block this rank owns.

    ``topk_ids`` stay in the global expert space (the no-dispatch convention of
    ``MoEPrepareAndFinalizeNoDPEPModular``), so the kernel needs the local
    block's offset instead of an ``expert_map`` -- it has no expert-map
    support. vLLM places a rank's experts contiguously, so rank ``r`` owns
    ``[r * L, (r + 1) * L)``, which is the layout FI assumes. In production
    this comes from ``moe_parallel_config``; here it is read back from
    ``expert_map`` once and cached by the caller.
    """
    if expert_map is None:
        assert local_num_experts == global_num_experts
        return 1, 0

    assert global_num_experts % local_num_experts == 0
    owned = torch.nonzero(expert_map >= 0, as_tuple=False).flatten()
    start = int(owned[0]) if owned.numel() else 0
    contiguous = (
        owned.numel() == local_num_experts
        and start % local_num_experts == 0
        and torch.equal(
            owned, torch.arange(start, start + local_num_experts, device=owned.device)
        )
        and torch.equal(
            expert_map[owned],
            torch.arange(
                local_num_experts, dtype=expert_map.dtype, device=owned.device
            ),
        )
    )
    if not contiguous:
        raise NotImplementedError(
            "flashinfer_moe_block_mm needs this rank's experts to form the "
            f"contiguous block [r * {local_num_experts}, "
            f"(r + 1) * {local_num_experts}); this expert_map does not."
        )
    return global_num_experts // local_num_experts, start // local_num_experts


def flashinfer_moe_block_mm(
    output: torch.Tensor,  # [num_tokens, H] bf16 (reduced layer output)
    hidden_states: torch.Tensor,  # [num_tokens, K] fp8 (already quantized)
    w1: torch.Tensor,  # [E, N, K] fp8 (gate_up, N = 2 * intermediate)
    w2: torch.Tensor,  # [E, H, intermediate] fp8 (down projection)
    w1_scale: torch.Tensor,  # [E, N // 128, K // 128] f32
    w2_scale: torch.Tensor,  # [E, H // 128, intermediate // 128] f32
    a1q_scale: torch.Tensor,  # [num_tokens, K // 128] f32
    topk_weights: torch.Tensor,  # [num_tokens, top_k] f32
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    expert_map: torch.Tensor | None,  # [global_num_experts] or None
    global_num_experts: int,
    workspace13: torch.Tensor,  # unused (the FI kernel fuses the whole layer)
    workspace2: torch.Tensor,  # unused
    clamp_limit: float | None = None,  # swiglu_limit of the gated activation
) -> None:
    """FlashInfer CUTLASS full-layer baseline for ``mega_moe_block_mm``.

    Mirrors ``FlashInferExperts.apply``'s ``use_deepseek_fp8_block_scale``
    branch -- ``quant_scales=[w1_scale, w2_scale]``, fp8 block-scaled weights,
    ``swiglu_limit`` for the clamp -- with one deliberate difference: vLLM
    passes unquantized activations (``input_sf=None``,
    ``expects_unquantized_inputs``) and lets the kernel quantize them itself,
    while this baseline passes the already-quantized fp8 activations plus their
    per-128-group scales through ``input_sf``, i.e. the same input the mega
    kernel receives. The W31 layout swap vLLM performs at load time is done on
    the first call (warmup) and cached.

    SM90 only: FlashInfer implements FP8 block scaling for Hopper ("FP8 block
    scaling not yet implemented for Blackwell"). Expert parallelism follows the
    no-dispatch convention (``MoEPrepareAndFinalizeNoDPEPModular``): ``topk_ids``
    stay in the global expert space and the kernel is told which contiguous
    local block this rank owns through ``ep_size``/``ep_rank`` -- the same
    information ``expert_map`` carries for the Triton path.
    """
    from vllm.platforms import current_platform

    if not current_platform.is_device_capability(90):
        raise NotImplementedError(
            "FlashInfer's FP8 block-scale MoE is Hopper (SM90) only."
        )
    try:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
            activation_to_flashinfer_type,
        )
        from vllm.utils.flashinfer import (
            flashinfer_cutlass_fused_moe,
            has_flashinfer_cutlass_fused_moe,
        )

        available = has_flashinfer_cutlass_fused_moe()
    except Exception as exc:  # noqa: BLE001 - flashinfer may fail to import
        raise NotImplementedError(f"FlashInfer unavailable: {exc}") from exc
    if not available:
        raise NotImplementedError("flashinfer_cutlass_fused_moe is unavailable.")

    state = _fi_state(
        w1, w2, w1_scale, w2_scale, expert_map, global_num_experts, clamp_limit
    )
    # Per-call work, as in FlashInferExperts.apply: adapt the routing dtype and
    # hand the kernel the activation blocks the mega kernel already consumed.
    _ = flashinfer_cutlass_fused_moe(
        input=hidden_states,
        token_selected_experts=topk_ids.to(torch.int),
        token_final_scales=topk_weights,
        fc1_expert_weights=state.w13,
        fc2_expert_weights=w2,
        fc1_expert_biases=None,
        fc2_expert_biases=None,
        swiglu_alpha=None,
        swiglu_beta=None,
        swiglu_limit=state.swiglu_limit,
        output=output,
        output_dtype=output.dtype,
        quant_scales=[state.w13_scale, state.fc2_scale],
        input_sf=a1q_scale if a1q_scale.is_contiguous() else a1q_scale.contiguous(),
        tp_size=1,
        tp_rank=0,
        ep_size=state.ep_size,
        ep_rank=state.ep_rank,
        activation_type=activation_to_flashinfer_type(MoEActivation.SILU),
        use_deepseek_fp8_block_scale=True,
        use_mxfp8_act_scaling=False,
        use_w4_group_scaling=False,
    )


# Maps a baseline name to a hand-written reference used as the ``custom``
# baseline. Entries keyed by a kernel name are that kernel's default custom
# baseline; the others are selected explicitly (e.g.
# ``--baseline flashinfer_moe_block_mm``). Each function shares the kernel's
# argument interface, so the kernel's input tuple is forwarded verbatim.
CUSTOM_BASELINE_FNS: dict[str, Callable] = {
    "block_scaled_mm": _block_scaled_mm,
    "fused_moe_block_mm": _moe_block_mm,
    "mega_moe_block_mm": triton_moe_block_mm,
}
