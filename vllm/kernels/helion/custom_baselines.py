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

from collections.abc import Callable

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


def _mega_moe_block_mm(
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
    from vllm.model_executor.layers.fused_moe.utils import _resize_cache
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

    # silu_and_mul + per-block fp8 quant of the intermediate.
    hq, hq_scale = ops.silu_and_mul_per_block_quant(
        gate_up.view(routed, N),
        group_size=128,
        quant_dtype=torch.float8_e4m3fn,
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


# Maps a Helion kernel name to a hand-written reference used as the ``custom``
# baseline. Each function shares the kernel's argument interface, so the
# kernel's input tuple is forwarded verbatim.
CUSTOM_BASELINE_FNS: dict[str, Callable] = {
    "block_scaled_mm": _block_scaled_mm,
    "fused_moe_block_mm": _moe_block_mm,
    "mega_moe_block_mm": _mega_moe_block_mm,
}
