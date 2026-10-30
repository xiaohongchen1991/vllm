# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Composite block-scaled MoE layer: one grouped GEMM run twice, with silu+quant
and the routed-weight combine between.

``moe_block_mm`` (below) is a generic routed grouped GEMM (fp8 activations
with per-128-group scales, 128x128 block-scaled weights, bf16 output), so the
composite runs it twice -- once as the gate_up projection (``w1``,
per-token activations gathered in-kernel) and once as the down projection
(``w2``, per-routed-row activations) -- with the production
``ops.silu_and_mul_per_block_quant`` in between operating on the full
[routed, 2 * intermediate] rows. Every call passes the real model ``top_k``;
the GEMM derives each stage's activation layout from ``aq.shape[0]``. The down
projection scales each output row by its routed slot's ``topk_weights``
coefficient in its epilogue (Triton's ``mul_routed_weight``), and
``ops.moe_sum`` reduces the weighted rows over top_k, so the composite models
the complete MoE layer end to end, mirroring ``TritonExperts.apply`` (and the
custom Triton baseline).

``permuted``, ``fuse_moe_sum`` and ``m_block_size`` are root tunables
(``run.tunable``).
``m_block_size`` is the alignment block size -- the routed rows of one
aligned block, shared by ``moe_align_block_size`` and both GEMM sites
(Triton fused_moe's ``BLOCK_SIZE_M``); its default keeps the
one-block-per-expert-segment layout of ``next_power_of_two(num_tokens)``.
``permuted`` selects the expert-major layout instead: the standalone
``moe_route`` kernel (called as its registered custom op, with its own
separately autotuned configs -- it is not a composite child site) produces
the aligned layout, the expert-major activation
rows, and ``inv_perm`` from one launch (replacing ``_align_routing`` +
``moe_permute``), and both GEMM stages gather
them through ``inv_perm``. GEMM1 also stores its gate_up output permuted
(keeping the intermediate contiguous per expert), while GEMM2 scatters its
weighted output to the routed slots' token-order rows, so the top-k combine
is always a contiguous ``moe_sum``. The autotuner explores both
layouts per shape and keeps the faster.
``fuse_moe_sum`` folds that combine into GEMM2's store instead: each
weighted routed row atomically accumulates into its token's row of the
[num_tokens, H] layer output (pre-zeroed by the kernel), replacing the
mm2_out staging cache and the separate ``moe_sum`` launch -- trading a
deterministic reduction for one fewer kernel and no [routed, H] round trip.
The gate_up projection never combines, so GEMM1 always passes False.

The interface is the fused ``fused_moe_block_mm`` interface plus
``topk_weights`` (inserted before ``topk_ids``), with the layer-level
[num_tokens, H] output in place of the fused kernel's unreduced
[num_tokens * top_k, H] one.

Autotuning searches the joint space -- the root tunables plus both GEMM call
sites' configs (sections ``children.moe_block_mm.0`` / ``.1``) -- with
helion's own ``LFBOTreeSearch``, measuring every candidate end to end::

    python scripts/autotune_helion_kernels.py --kernels mega_moe_block_mm

The composite owns the routing and the scratch buffers: it runs
``moe_align_block_size`` once at the root-tuned ``m_block_size`` (both GEMMs
consume that alignment) and resizes the preallocated workspace
caches like the Triton baseline -- ``workspace2`` to the bf16 gate_up cache
[num_tokens, top_k, N] (TritonExperts' intermediate_cache1) and ``workspace13``
to the bf16 weighted GEMM2 output [num_tokens, top_k, H]. At
``m_block_size == 1`` the alignment is skipped: every routed slot is its own
block, so the GEMMs take the dense layout (``sorted_token_ids=None``) and
derive slot = block in-kernel (Triton fused_moe's
``naive_block_assignment``), with ``expert_ids`` the flat ``topk_ids`` view.
The routing also produces the actual padded routed-slot count
(``num_tokens_post_padded``, a [1] device tensor from moe_align /
moe_route / the dense-path synthesis): the align producers -1-fill the
``expert_ids`` tail beyond it, so the GEMMs' ``e >= 0`` branch skips the
worst-case tail blocks, and the count gates the off-rank zero-fill to
blocks whose rows moe_sum actually reads. Everything
runs on-device with no host sync, so the composite is **CUDA-graph
compatible**.
"""

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

from vllm import _custom_ops as ops

# Imported for its registration side effect: the permuted branch below calls
# the registered custom op, which must exist regardless of ops import order.
from vllm.kernels.helion.inputs import CaseInputs
from vllm.kernels.helion.ops import moe_route as _moe_route_module  # noqa: F401
from vllm.kernels.helion.register import register_kernel
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
    fused_topk,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
    swiglu_limit_func,
)

logger = init_logger(__name__)

GROUP_K = 128
GROUP_N = 128

# Masked atomic accumulate into the output rows. Mirrors ``hl.atomic_add``,
# which has no mask argument: masked lanes are skipped outright instead of being
# redirected to a valid row, so a padded lane's sentinel index (one row past the
# output) is never issued and no lane contends on the redirect row.
# ``{p}`` output base pointer, ``{r}`` element row offset, ``{n}`` column tile
# index, ``{v}`` values, ``{m}`` lane mask.
_MASKED_ATOMIC_ADD = (
    "tl.atomic_add({p} + {r}[:, None] + {n}[None, :], {v}, mask={m}, sem='relaxed')"
)


def _align_sizes(
    num_tokens: int, top_k: int, E: int, align_block_size: int
) -> tuple[int, int]:
    numel = num_tokens * top_k
    max_padded = numel + E * (align_block_size - 1)
    if numel < E:
        max_padded = min(numel * align_block_size, max_padded)
    num_blocks = (max_padded + align_block_size - 1) // align_block_size
    return max_padded, num_blocks


# The composite's child GEMM: a plain helion kernel (the composite owns tuning
# and accuracy). static_shapes=False keeps the align outputs' runtime sizes
# (max_padded / num_blocks) unspecialized, as in the fused kernel. Codegen-shape
# tunables are pinned by the composite's stage-1 scheduling (composite.py), not
# here, so this kernel carries no autotune overrides of its own.
@helion.kernel(
    static_shapes=False,
    ignore_warnings=[
        helion.exc.TensorOperationInWrapper,
        helion.exc.ProcessGroupNameNotFound,
    ],
)
def moe_block_mm(
    output: torch.Tensor,  # [routed, N] (out; [num_tokens, N] when fuse_moe_sum)
    aq: torch.Tensor,  # [M, K] fp8; M = num_tokens or num_tokens * top_k
    aq_scale: torch.Tensor,  # [M, K // 128]
    w: torch.Tensor,  # [E, N, K] fp8
    w_scale: torch.Tensor,  # [E, N // 128, K // 128]
    sorted_token_ids: torch.Tensor | None,  # align output, int32; None = dense layout
    expert_ids: torch.Tensor,  # align output, int32
    num_tokens: int,
    top_k: int,  # real model top_k (routed experts per token)
    align_block_size: hl.constexpr,  # align block size: routed rows per aligned block
    split_k: hl.constexpr,
    num_tokens_post_padded: torch.Tensor,  # [1] int32: padded routed-slot count
    topk_weights: torch.Tensor | None = None,  # [num_tokens, top_k] routed weights
    inv_perm: torch.Tensor | None = None,  # [routed] slot -> permuted row
    fuse_moe_sum: hl.constexpr = False,  # accumulate into the [num_tokens, N] output
    clamp_limit: hl.constexpr | None = None,  # silu/up clamp of the fused epilogue
    output_scale: torch.Tensor | None = None,  # [routed, N // 2 // 128] fp32 (out)
) -> None:
    M, K = aq.shape
    E, N = w.shape[0], w.shape[1]
    routed = num_tokens * top_k

    hl.specialize(M)
    hl.specialize(K)
    hl.specialize(E)
    hl.specialize(N)
    hl.specialize(top_k)

    # Each activation row serves slots_per_row consecutive routed slots: top_k
    # when aq holds one row per token (the unpermuted gate_up projection), 1
    # when one row per routed slot (everything else). num_tokens, top_k and M
    # are all specialized, so this folds to a compile-time constant.
    slots_per_row = routed // M
    assert slots_per_row >= 1 and slots_per_row * M == routed

    assert w.shape == (E, N, K)
    if fuse_moe_sum:
        # Fused combine (the down projection only): output is the
        # [num_tokens, N] layer output; each weighted routed row accumulates
        # into its token's row.
        assert topk_weights is not None
        assert output.shape == (num_tokens, N)
    else:
        # The gate_up projection stores the full gate_up row, or -- with the
        # fused silu+block-quant epilogue -- one column per intermediate
        # element.
        proj_N = N // 2 if output_scale is not None else N
        assert output.shape == (routed, proj_N)
    assert aq.dtype == w.dtype
    assert output.dtype.is_floating_point
    assert aq.stride(1) == 1
    assert output.stride(1) == 1
    if fuse_moe_sum or split_k > 1:
        # The masked atomic below builds its pointer by hand, so the row stride
        # must be the projection width (the fused act+quant epilogue implies
        # split_k == 1, so its [routed, N // 2] output never reaches it).
        assert output.shape[1] == N
        assert output.stride(0) == N
    # aq_scale may be transposed (e.g. the ue8m0 activation scales arrive as
    # [K // 128, num_tokens].t()); loads index it by (row, group) strides.
    assert w.stride(2) == 1
    assert num_tokens >= 1
    assert top_k >= 1

    # permuted mode (inv_perm is not None): aq/aq_scale hold moe_permute's
    # expert-major rows (one row per routed slot, no padding), gathered
    # through inv_perm (routed slot -> permuted row). The gate_up projection
    # (topk_weights is None) also stores through inv_perm, keeping the
    # intermediate permuted so downstream stages read contiguous per-expert
    # segments; the down projection scatters its weighted output to the routed
    # slots' token-order rows, so the combine is a contiguous moe_sum.
    if inv_perm is not None:
        assert slots_per_row == 1
        assert inv_perm.ndim == 1
        assert inv_perm.numel() == routed

    # Optional routed weights (the MoE combine coefficients), one value per
    # routed slot, applied per output row in the epilogue like Triton's
    # mul_routed_weight. Only the down projection passes them (slots_per_row
    # == 1, where the routed slot offs_token is also the store's row index in
    # both layouts).
    apply_weights = topk_weights is not None
    if apply_weights:
        assert slots_per_row == 1
        # Flat per-slot view under a distinct name: the branch must not rebind
        # topk_weights to a different rank. reshape(-1) always yields a
        # stride-1 flat tensor (view or copy).
        topk_weights_flat = topk_weights.reshape(-1)
        assert topk_weights_flat.numel() == routed
        assert topk_weights_flat.dtype.is_floating_point

    assert expert_ids.ndim == 1
    assert expert_ids.dtype == torch.int32
    if sorted_token_ids is not None:
        assert sorted_token_ids.ndim == 1
        assert sorted_token_ids.dtype == torch.int32

    # The routing's actual padded routed-slot count: stays on the device (the
    # grid is sized from the worst-case buffers; the count only prunes
    # in-kernel), so the composite keeps CUDA-graph compatibility.
    assert num_tokens_post_padded.dtype == torch.int32
    assert num_tokens_post_padded.numel() == 1

    assert aq_scale.ndim == 2 and w_scale.ndim == 3
    assert aq_scale.dtype == w_scale.dtype and aq_scale.is_floating_point()

    num_group_k = aq_scale.shape[1]
    num_group_n = w_scale.shape[1]
    assert w_scale.shape == (E, num_group_n, num_group_k)
    assert aq_scale.shape == (M, num_group_k)

    assert K % num_group_k == 0
    assert N % num_group_n == 0

    # aq_scale group shape must be [1, 128], w_scale group shape [128, 128].
    group_k = K // num_group_k
    group_n = N // num_group_n
    assert group_k == 128 and group_n == 128

    hl.specialize(num_group_n)
    hl.specialize(num_group_k)

    hl.specialize(group_n)
    hl.specialize(group_k)

    acc_dtype = torch.float32 if aq.is_floating_point() else torch.int32
    out_dtype = output.dtype

    # Routing contract: the caller aligned at block_size = align_block_size (passed
    # in and specialized), so each aligned block holds routed rows of exactly
    # one expert (real rows at the head, padding after) and expert_ids holds
    # the active experts in ascending order followed by a -1 tail. The grid
    # covers the worst-case buffers (num_blocks = expert_ids.size(0), the
    # tight host-computable bound); num_tokens_post_padded, the actual
    # per-batch count, stays on device -- the -1 tail already skips the
    # beyond-count blocks, and the count gates the zero-fill branch.
    # sorted_token_ids=None selects the dense layout instead
    # (Triton fused_moe's naive_block_assignment): block b is routed slot b
    # with no padding, the slot mapping is derived in-kernel, and expert_ids is
    # the caller's flat topk_ids view.
    if sorted_token_ids is None:
        assert align_block_size == 1
        max_padded = routed
    else:
        max_padded = sorted_token_ids.size(0)
    num_blocks = expert_ids.size(0)
    assert num_blocks <= routed
    assert num_blocks * align_block_size >= max_padded

    swap_ab = hl.register_tunable("swap_ab", BooleanFragment())

    k_block_size = helion.next_power_of_2(helion.cdiv(K, split_k))
    # split_k partials and the fused moe_sum store both accumulate, so both
    # need a zeroed output.
    if split_k > 1 or (apply_weights and fuse_moe_sum):
        output.zero_()

    # Passing a scale output asks for the fused silu+block-quant epilogue; it is
    # taken only when a program owns a whole 128-wide quantization group and K
    # is accumulated in full (split-K partials cannot be quantized
    # independently). Both arms of the flag are traced, so everything they
    # produce is unified on out_dtype.
    if output_scale is not None:
        assert not apply_weights
        assert split_k == 1

    # The clamp is part of the fused silu epilogue (MoEActivation.SILU with a
    # swiglu_limit); the unfused path applies it before quantizing instead.
    if clamp_limit is not None:
        assert output_scale is not None

    fuse_act_quant = output_scale is not None
    adjusted_N = N // 2 if fuse_act_quant else N
    min_n_block_size = 128 if fuse_act_quant else 1

    qtype_max = 448.0
    qtype_min = -qtype_max
    min_scaling_factor = 1.0 / (qtype_max * 512.0)

    m_block_size = hl.register_block_size(align_block_size)
    n_block_size = hl.register_block_size(min_n_block_size, adjusted_N)

    for tile_e, tile_m, tile_n, outer_k in hl.tile(
        [num_blocks, align_block_size, adjusted_N, K],
        block_size=[1, m_block_size, n_block_size, k_block_size],
    ):
        blk = tile_e.begin
        ntp = num_tokens_post_padded[0]

        if blk * align_block_size < ntp:
            e = expert_ids[blk]
            rows = blk * align_block_size + tile_m.index
            if sorted_token_ids is None:
                # Dense layout: block b is routed slot b, derived
                # arithmetically -- no sorted-ids load, no padding.
                offs_token = rows
                token_mask = rows < routed
            else:
                row_mask = rows < max_padded
                offs_token = hl.load(sorted_token_ids, [rows], extra_mask=row_mask)
                token_mask = row_mask & (offs_token < routed)

            if e >= 0:
                # The permuted / unpermuted split keys on inv_perm's presence:
                # None checks resolve at trace time, so only the taken layout is
                # traced and the compiled code has no branch on the mode at all.
                if inv_perm is not None:
                    # aq is in moe_permute's expert-major order: permuted row
                    # inv_perm[s] holds routed slot s, so the activation gather
                    # goes through inv_perm (masked lanes load 0). The gate_up
                    # projection stores permuted as well (contiguous per expert);
                    # the down projection lands on the routed slot's token-order
                    # row so the combine is a contiguous moe_sum.
                    # Padded lanes hold the sorted-ids sentinel (one past the
                    # table); the load is masked, so the index is never used.
                    inv_rows = hl.load(inv_perm, [offs_token], extra_mask=token_mask)
                    aq_rows = inv_rows
                    store_rows = offs_token if apply_weights else inv_rows
                else:
                    # Routed slot offs_token reads activation row offs_token //
                    # slots_per_row (the token's row for the per-token gate_up
                    # layout; the identity for per-slot activations) and writes
                    # output row offs_token. Its loads are masked too, so the
                    # padded lanes' sentinel index is never dereferenced.
                    aq_rows = offs_token // slots_per_row
                    store_rows = offs_token

                if fuse_moe_sum:
                    # The combine target is the token's row in the
                    # [num_tokens, N] layer output.
                    store_rows = offs_token // top_k

                acc = hl.zeros([tile_m, tile_n], torch.float32)

                if fuse_act_quant:
                    acc2 = hl.zeros([tile_m, tile_n], torch.float32)

                # Keep block_size = group_k for the K dimension so the block scale
                # is constant across each inner tile.
                for tile_k in hl.tile(outer_k.begin, outer_k.end, block_size=group_k):
                    aq_blk = hl.load(
                        aq, [aq_rows, tile_k], extra_mask=token_mask[:, None]
                    )
                    w_blk = w[e, tile_n, tile_k]
                    if fuse_act_quant:
                        # Up half, adjusted_N columns past the gate half. The
                        # offset index is masked explicitly: a padded lane of
                        # the last n tile (tile_n.index >= adjusted_N) would
                        # otherwise read at or past N.
                        w2_blk = hl.load(
                            w,
                            [e + hl.arange(1), tile_n.index + adjusted_N, tile_k],
                            extra_mask=(tile_n.index + adjusted_N < N)[None, :, None],
                        ).squeeze(0)

                    if swap_ab:
                        acc_blk = (
                            hl.dot(w_blk, aq_blk.t(), out_dtype=acc_dtype)
                            .t()
                            .to(torch.float32)
                        )
                        if fuse_act_quant:
                            acc2_blk = (
                                hl.dot(w2_blk, aq_blk.t(), out_dtype=acc_dtype)
                                .t()
                                .to(torch.float32)
                            )
                    else:
                        acc_blk = hl.dot(aq_blk, w_blk.t(), out_dtype=acc_dtype).to(
                            torch.float32
                        )
                        if fuse_act_quant:
                            acc2_blk = hl.dot(
                                aq_blk, w2_blk.t(), out_dtype=acc_dtype
                            ).to(torch.float32)

                    gk_idx = tile_k.id
                    aq_scale_blk = hl.load(
                        aq_scale,
                        [aq_rows, gk_idx + hl.arange(1)],
                        extra_mask=token_mask[:, None],
                    )
                    w_scale_blk = w_scale[e, tile_n.index // group_n, gk_idx][None, :]
                    if fuse_act_quant:
                        # 1-D index and mask (e and gk_idx are scalars), then
                        # the same [None, :] promotion as the gate half.
                        w2_scale_blk = hl.load(
                            w_scale,
                            [
                                e + hl.arange(1),
                                (tile_n.index + adjusted_N) // group_n,
                                gk_idx + hl.arange(1),
                            ],
                            extra_mask=(
                                (tile_n.index + adjusted_N) // group_n < num_group_n
                            )[None, :, None],
                        ).squeeze(-1)

                    acc = acc + (aq_scale_blk * w_scale_blk) * acc_blk
                    if fuse_act_quant:
                        acc2 = acc2 + (aq_scale_blk * w2_scale_blk) * acc2_blk

                if fuse_act_quant:
                    # silu(gate) * up, then per-128-group fp8 quantization --
                    # the epilogue form of ops.silu_and_mul_per_block_quant
                    # (scale = amax / 448 clamped, q = act / scale, RN cast).
                    # act is [tile_m, n_block_size] with n_block_size a power
                    # of two >= 128, so the grouped reshape yields one 128-wide
                    # group per (row, group) pair and the reduction keeps that
                    # [tile_m, groups] shape.
                    if clamp_limit is None:
                        act = acc * torch.sigmoid(acc) * acc2
                    else:
                        # MoEActivation.SILU with a swiglu_limit, as
                        # swiglu_limit_func applies it: the gate is clamped from
                        # above only, the up half symmetrically, and both before
                        # the nonlinearity so the group scale quantizes the
                        # clamped activation.
                        gate = torch.clamp_max(acc, clamp_limit)
                        act = (
                            gate
                            * torch.sigmoid(gate)
                            * torch.clamp(acc2, -clamp_limit, clamp_limit)
                        )
                    num_group_tile = n_block_size // 128
                    act_grouped = act.reshape(m_block_size, num_group_tile, 128)
                    s = torch.amax(torch.abs(act_grouped), dim=-1).to(torch.float32)
                    s = (s * (1.0 / qtype_max)).clamp(min=min_scaling_factor)
                    q = (act_grouped / s[:, :, None]).clamp(qtype_min, qtype_max)
                    acc = q.reshape(act.shape)
                    group_idx = tile_n.begin // 128 + hl.arange(num_group_tile)
                    hl.store(
                        output_scale,
                        [store_rows, group_idx],
                        s,
                        extra_mask=token_mask[:, None]
                        & (group_idx < (adjusted_N // 128))[None, :],
                    )

                # Routed weight per output row (Triton's mul_routed_weight),
                # indexed by the routed slot (aq_rows would be wrong in the
                # permuted layout, where it holds permuted rows). Split-K partials
                # are disjoint K-slices of the same output row, so weighting each
                # partial equals weighting the total: sum_k(w * P_k) == w * P.
                if apply_weights:
                    # Masked load: padded lanes' sentinel index is never read.
                    weight_blk = hl.load(
                        topk_weights_flat, [offs_token], extra_mask=token_mask
                    )
                    acc = acc * weight_blk[:, None]

                if split_k == 1 and not fuse_moe_sum:
                    hl.store(
                        output,
                        [store_rows, tile_n],
                        acc.to(out_dtype),
                        extra_mask=token_mask[:, None],
                    )
                else:
                    # Atomic accumulate: split-K partials are disjoint K-slices
                    # of the same output row, and the folded top_k combine
                    # (fuse_moe_sum) lands several rows on one token, so both
                    # need read-modify-write (the output was pre-zeroed above).
                    # Masked lanes must not be issued at all: their index is the
                    # padding sentinel, which is one row past the output, and
                    # redirecting them to a valid row (the previous
                    # ``torch.where(..., 0)``) serializes every padded lane of
                    # every block on one address. ``hl.atomic_add`` has no mask
                    # parameter, so emit the Triton atomic directly -- it masks
                    # the address, not just the value.
                    hl.inline_triton(
                        _MASKED_ATOMIC_ADD,
                        args={
                            "p": output,
                            "r": store_rows * N,  # element row offset
                            "n": tile_n.index,
                            "v": acc.to(out_dtype),
                            "m": token_mask[:, None],
                        },
                        output_like=None,
                    )
            elif apply_weights and split_k == 1 and not fuse_moe_sum:
                # Only the store row is needed here: the down projection always
                # writes the routed slot's token-order row (both layouts), and no
                # activation is gathered.
                store_rows = offs_token

                # Off-rank block inside the actual count: zero its output rows
                # (Triton's write_zeros_to_output) so a combine that reads every
                # slot sees defined zeros. Both callers now route compactly
                # (off-rank slots get no block), so this only fires for a
                # layout that keeps them; blocks beyond the count are excluded
                # either way, their rows being padding markers. split_k > 1
                # pre-zeroes the whole output, and the fused combine skips
                # this too: an atomic zero-contribution would be a no-op.
                hl.store(
                    output,
                    [store_rows, tile_n],
                    hl.zeros([tile_m, tile_n], out_dtype),
                    extra_mask=token_mask[:, None],
                )


def _align_routing(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    global_num_experts: int,
    block_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Align the routing once for both GEMMs at ``block_size``, so each
    aligned block holds routed rows of exactly one expert.

    The layout is compact (``ignore_invalid_experts=True``): a slot routed to
    another rank gets no block, so both GEMMs prune their grids to the blocks
    this rank computes. Those slots' staged rows stay unwritten, which the
    pad-aware ``ops.moe_sum`` skips at the combine.

    At block_size == 1 (and no expert_map) the alignment is skipped: every
    routed slot is its own block, so the dense layout (sorted_token_ids=None,
    expert = topk_ids[p]) satisfies the stage contract, and the padded count
    is synthesized like ``_prepare_expert_assignment``'s naive path. The GEMM
    only reads ``expert_ids[blk]``, so block order does not matter.
    """
    if block_size == 1 and expert_map is None:
        return (
            None,
            topk_ids.view(-1),
            torch.full(
                (1,),
                topk_ids.numel() * block_size,
                dtype=torch.int32,
                device=topk_ids.device,
            ),
        )

    return moe_align_block_size(
        topk_ids,
        block_size,
        global_num_experts,
        expert_map,
        ignore_invalid_experts=True,
    )


def _simulate_routing(
    num_tokens: int,
    E: int,
    top_k: int,
    routing_seed: int = 0,
    scale_dtype: torch.dtype = torch.float32,
    fixed_seed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One synthetic routing draw: top_k expert ids and their routed weights.

    Gaussian gating logits (i.i.d. across experts, i.e. uniform expected load)
    pushed through the production router, ``fused_topk``. With ``fixed_seed``
    the draw is seeded per shape and draw so autotuning is reproducible;
    otherwise it comes from the global RNG. Runs eagerly so it produces
    concrete values under the autotuner's FakeTensorMode.

    Ids are in the global expert space; the caller decides which rank the case
    models (see ``_local_rank_for``).
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    with unset_fake_temporarily():
        gen = (
            torch.Generator(device="cuda").manual_seed(
                num_tokens * 1000 + E * 10 + top_k + routing_seed * 7919
            )
            if fixed_seed
            else None
        )
        gating = torch.randn(
            num_tokens, E, dtype=scale_dtype, device="cuda", generator=gen
        )
        topk_weights, topk_ids, _ = fused_topk(gating, gating, top_k, renormalize=True)
    return topk_ids, topk_weights


def _local_rank_for(topk_ids: torch.Tensor, num_local_experts: int) -> int:
    """The expert-parallel rank this draw loads most heavily.

    A step's latency is set by its busiest rank: the others finish their expert
    kernel early and wait in the finalize collective, so tuning for the busiest
    rank approximates the step time. It also guarantees the case has work -- at
    small batches most ranks get no slot at all (34% at EP=8, one token), and a
    no-op rank's timing says nothing about any config.

    ``topk_ids`` comes from ``_simulate_routing``'s fake-mode escape, so this
    runs under the same escape: the ids are concrete tensors even while the
    autotuner traces the input generator under FakeTensorMode.
    """
    from torch._subclasses.fake_tensor import unset_fake_temporarily

    with unset_fake_temporarily():
        slots_per_rank = torch.bincount(
            topk_ids.flatten().to(torch.int64) // num_local_experts
        )
        return int(slots_per_rank.argmax())


def _replace_routing(
    args: tuple[Any, ...], topk_weights: torch.Tensor, topk_ids: torch.Tensor
) -> tuple[Any, ...]:
    """Reuse a case's tensors with a different routing draw (multi-input)."""
    (
        output,
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        a1q_scale,
        _topk_weights,
        _topk_ids,
        expert_map,
        global_num_experts,
        workspace13,
        workspace2,
        clamp_limit,
    ) = args
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
        clamp_limit,
    )


def _generate_input(
    E: int,
    K: int,
    N: int,
    num_tokens: int,
    top_k: int,
    fp8: torch.dtype,
    ep_size: int = 1,
    clamp_limit: float | None = None,
    routing_seed: int = 0,
    fixed_seed: bool = True,
) -> tuple[Any, ...]:
    assert E % ep_size == 0
    num_local_experts = E // ep_size

    scale_dtype = torch.float32
    out_dtype = torch.bfloat16
    scale = 1.0 / math.sqrt(K)

    num_group_k = K // GROUP_K
    num_group_n = N // GROUP_N

    # GEMM1 projects to the gate_up tensor [*, N]; silu_and_mul folds it to the
    # [*, N // 2] intermediate; GEMM2 (down projection) maps the intermediate
    # back to the model hidden dim H == K, producing the layer's [num_tokens, H]
    # output (the weighted routed rows reduced over top_k).
    intermediate = N // 2
    H = K
    num_group_i = intermediate // GROUP_N
    num_group_h = H // GROUP_N

    # topk_ids live in the global expert space, as a TP+EP rank receives them;
    # the routed combine weights come from the same seeded draw.
    topk_ids, topk_weights = _simulate_routing(
        num_tokens, E, top_k, routing_seed, scale_dtype, fixed_seed
    )

    scale2 = 1.0 / math.sqrt(intermediate)
    hidden_states = (scale * (0.5 + torch.rand(num_tokens, K, device="cuda"))).to(fp8)
    w1 = (scale * (0.5 + torch.rand(num_local_experts, N, K, device="cuda"))).to(fp8)
    w2 = (
        scale2 * (0.5 + torch.rand(num_local_experts, H, intermediate, device="cuda"))
    ).to(fp8)
    # Layer-level output: the top_k-weighted rows reduced over top_k.
    output = torch.zeros((num_tokens, H), dtype=out_dtype, device="cuda")

    a1q_scale = 0.5 + torch.rand(
        num_group_k, num_tokens, dtype=scale_dtype, device="cuda"
    )
    a1q_scale = _ceil_to_ue8m0(a1q_scale).t()

    # Weight scales: w1 [E, num_group_n, num_group_k], w2 [E, num_group_h,
    # num_group_i].
    w1_scale = 0.5 + torch.rand(
        num_local_experts, num_group_n, num_group_k, dtype=scale_dtype, device="cuda"
    )
    w1_scale = _ceil_to_ue8m0(w1_scale)
    w2_scale = 0.5 + torch.rand(
        num_local_experts, num_group_h, num_group_i, dtype=scale_dtype, device="cuda"
    )
    w2_scale = _ceil_to_ue8m0(w2_scale)

    # A TP+EP rank's view: the rank this draw loads most heavily owns the local
    # experts (reindexed from 0), every other expert is -1. Off-rank slots never
    # get blocks, so GEMM2 zero-fills their output rows -- the rank's partial
    # sum.
    if ep_size == 1:
        expert_map = None
    else:
        local_rank = _local_rank_for(topk_ids, num_local_experts)
        expert_map = torch.full((E,), -1, dtype=torch.int32, device="cuda")
        expert_map[
            local_rank * num_local_experts : (local_rank + 1) * num_local_experts
        ] = torch.arange(num_local_experts, dtype=torch.int32, device="cuda")
    global_num_experts = E

    # Scratch workspaces shared by the Helion kernel and the custom baselines
    # (the same tuple is forwarded to both): workspace2 stages the bf16 gate_up
    # cache [num_tokens, top_k, N] and workspace13 the bf16 GEMM2 output
    # [num_tokens, top_k, H] (TritonExperts' intermediate_cache1 / _3 roles).
    workspace13 = torch.empty(
        (num_tokens, top_k, max(intermediate, H)), dtype=out_dtype, device="cuda"
    )
    workspace2 = torch.empty(
        (num_tokens, top_k, max(N, H)), dtype=out_dtype, device="cuda"
    )
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
        clamp_limit,
    )


def generate_inputs() -> dict[CaseKey, CaseInputs]:
    # Hardcoded switch: with a fixed seed the weights, scales and routing draws
    # are identical on every run, so autotuning and benchmarking reproduce. Set
    # it to False to draw fresh inputs per call, e.g. to check that a tuned
    # config is not fitted to one sample.
    fixed_seed = True

    # The Helion kernel is autotuned per shape.
    # num_tokens_list follows the cudagraph_capture_sizes pattern but is capped
    # here to cover only small values.
    num_tokens_list = [1, 2, 4, 8, 16, 24, 32]

    fp8: torch.dtype = current_platform.fp8_dtype()

    # (E, top_k, EP, (K, N), clamp_limit) entries. E is the model's global
    # expert count and EP the deployment's expert-parallel size: the rank under
    # test owns E // EP experts at full intermediate width (N = 2 *
    # intermediate). clamp_limit is the model config's ``swiglu_limit`` -- the
    # MoEActivation.SILU clamp the fused epilogue applies -- or None when the
    # config sets none.
    # - Qwen3-30B-A3B, 1x H100 (~34GB fp8 weights), TP=1/EP=1: 128 experts,
    #   hidden 2048, intermediate 768. silu, no swiglu_limit.
    # - Qwen3-235B-A22B, 4x H100 (~240GB fp8 weights), TP=4/EP=4: 128
    #   experts, hidden 4096, intermediate 1536. silu, no swiglu_limit.
    # - GLM-5.3, 8x H200 (~745GB fp8 weights, the minimum single-node fit),
    #   TP=8/EP=8: 256 experts, hidden 6144, intermediate 2048. silu, no
    #   swiglu_limit.
    # - GLM-5.3-Flash, 8x H100: 288 experts, hidden 4096, intermediate 2048.
    #   silu with swiglu_limit 10.0, so the fused epilogue must clamp.
    e_topk_ep_kn_clamp_list: list[
        tuple[int, int, int, tuple[int, int], float | None]
    ] = [
        # (128, 8, 1, (2048, 1536), None),  # Qwen3-30B-A3B
        # (128, 8, 4, (4096, 3072), None),  # Qwen3-235B-A22B
        (256, 8, 8, (6144, 4096), None),  # GLM-5.3
        # (256, 8, 1, (6144, 512), None),  # GLM-5.3
        # (288, 8, 8, (4096, 4096), 10.0),  # GLM-5.3-Flash
    ]

    inputs = {}
    for num_tokens, (E, top_k, ep_size, (K, N), clamp_limit) in product(
        num_tokens_list, e_topk_ep_kn_clamp_list
    ):
        # Tune over two routing draws: a single draw is a poor estimate of the
        # padding/occupancy a config sees because per-expert load varies between
        # draws, and two independent draws take that sampling noise below the
        # gap between candidates.
        base = _generate_input(
            E,
            K,
            N,
            num_tokens,
            top_k,
            fp8,
            ep_size,
            clamp_limit,
            fixed_seed=fixed_seed,
        )
        n_routing = 4
        arg_sets = [base]
        for seed in range(1, n_routing):
            topk_ids, topk_weights = _simulate_routing(
                num_tokens, E, top_k, seed, fixed_seed=fixed_seed
            )
            arg_sets.append(_replace_routing(base, topk_weights, topk_ids))
        inputs[
            CaseKey(
                {
                    "E": E,
                    "EP": ep_size,
                    "K": K,
                    "N": N,
                    "topk": top_k,
                    "num_tokens": num_tokens,
                    "in_dtype": str(fp8),
                }
            )
        ] = tuple(arg_sets)

    return inputs


_pick_cache: dict[tuple[int, int, int, int, int, int, str], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned joint config for the given input shape.

    Configs are matched within the runtime input dtype and ``top_k``. E
    (global experts), EP, K/N are picked by closest match. ``num_tokens`` is
    bucketed to the smallest tuned value >= runtime.
    """

    if not config_keys:
        return None

    output, hidden_states, w1, w2, w1_s, w2_s, a1q_s, topk_w, topk_ids = args[:9]
    global_num_experts = args[10]

    num_tokens, K = hidden_states.shape
    top_k = topk_ids.shape[1]
    # w1 holds this rank's local experts; EP follows from the global count.
    num_local_experts, N = w1.shape[0], w1.shape[1]
    ep_size = global_num_experts // num_local_experts
    in_dtype = str(hidden_states.dtype)

    cache_key = (global_num_experts, ep_size, K, N, top_k, num_tokens, in_dtype)
    if cache_key in _pick_cache:
        return _pick_cache[cache_key]

    candidates: list[CaseKey] = []
    for key in config_keys:
        if key.is_default():
            continue
        if not all(
            k in key for k in ("E", "EP", "K", "N", "num_tokens", "topk", "in_dtype")
        ):
            continue
        if key["in_dtype"] != in_dtype:
            continue
        candidates.append(key)

    if not candidates:
        _pick_cache[cache_key] = None
        return None

    def sort_key(key: CaseKey) -> tuple[int, int, int, int, int, int, int]:
        nt = key["num_tokens"]
        covers = 0 if nt >= num_tokens else 1
        return (
            abs(key["topk"] - top_k),
            abs(key["EP"] - ep_size),
            abs(key["E"] - global_num_experts),
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
    clamp_limit: float | None = None,  # swiglu_limit of the gated activation
) -> None:
    return


def _silu_mul_block_quant(
    gate_up: torch.Tensor,  # [routed, N] bf16, gate || up
    clamp_limit: float | None,
    act_out: torch.Tensor | None = None,  # [routed, N // 2] bf16 scratch
) -> tuple[torch.Tensor, torch.Tensor]:
    """silu(gate) * up, then per-128-group fp8 quantization of each row.

    Without a clamp limit this is the fused CUDA op. With one it takes the two
    steps ``TritonExperts`` falls back to -- ``swiglu_limit_func`` into bf16,
    then the block quantization -- because the fused op has no clamp argument.
    """
    if clamp_limit is None:
        return ops.silu_and_mul_per_block_quant(
            gate_up, group_size=GROUP_K, quant_dtype=torch.float8_e4m3fn
        )
    routed, N = gate_up.shape
    act = (
        torch.empty((routed, N // 2), dtype=gate_up.dtype, device=gate_up.device)
        if act_out is None
        else act_out
    )
    swiglu_limit_func(act, gate_up, clamp_limit)
    return moe_kernel_quantize_input(
        act, None, torch.float8_e4m3fn, False, [GROUP_K, GROUP_K]
    )


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
    clamp_limit: float | None = None,  # swiglu_limit of the gated activation
) -> None:
    """Eager layer reference: GEMM1 reference, silu+quant op, weighted GEMM2
    reference, then moe_sum.

    Mode-agnostic (the permuted kernel computes the same layer values in a
    different row order, so the unpermuted reference validates both).
    """
    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    N = w1.shape[1]
    H = w2.shape[1]
    routed = num_tokens * top_k

    # GEMM1 reference: bf16 gate_up in routed-row order. The reference aligns
    # at its own block size; the layer output is invariant to the layout.
    gate_up = torch.empty((routed, N), dtype=output.dtype, device=hidden_states.device)
    align_block_size = helion.next_power_of_2(num_tokens)
    sorted_token_ids, expert_ids, _ = _align_routing(
        topk_ids, expert_map, global_num_experts, align_block_size
    )
    if sorted_token_ids is None:
        # Dense layout (align_block_size == 1): block b is routed slot b.
        # Materialize the identity mapping so the loops below stay
        # layout-agnostic.
        sorted_token_ids = torch.arange(
            routed, dtype=torch.int32, device=hidden_states.device
        )
    max_padded = sorted_token_ids.numel()
    a_deq = hidden_states.to(torch.float32) * a1q_scale.to(
        torch.float32
    ).repeat_interleave(GROUP_K, dim=1)
    for blk in range(expert_ids.numel()):
        e = int(expert_ids[blk])
        if e < 0:
            continue
        rows = torch.arange(align_block_size, device=sorted_token_ids.device)
        rows = rows + blk * align_block_size
        rows = rows[rows < max_padded]
        offs = sorted_token_ids[rows].to(torch.long)
        offs = offs[offs < routed]
        if offs.numel() == 0:
            continue
        w1_e = w1[e].to(torch.float32)
        w1_s = (
            w1_scale[e]
            .to(torch.float32)
            .repeat_interleave(GROUP_N, dim=0)
            .repeat_interleave(GROUP_K, dim=1)
        )
        gate_up[offs] = (a_deq[offs // top_k] @ (w1_e * w1_s).t()).to(output.dtype)
    hq, hq_scale = _silu_mul_block_quant(gate_up, clamp_limit)

    # GEMM2 reference: weighted routed rows, then reduce over top_k.
    hq_deq = hq.to(torch.float32) * hq_scale.to(torch.float32).repeat_interleave(
        GROUP_K, dim=1
    )
    w_flat = topk_weights.reshape(-1).to(torch.float32)
    out_rows = torch.zeros((routed, H), dtype=output.dtype, device=hidden_states.device)
    for blk in range(expert_ids.numel()):
        e = int(expert_ids[blk])
        if e < 0:
            continue
        rows = torch.arange(align_block_size, device=sorted_token_ids.device)
        rows = rows + blk * align_block_size
        rows = rows[rows < max_padded]
        offs = sorted_token_ids[rows].to(torch.long)
        offs = offs[offs < routed]
        if offs.numel() == 0:
            continue
        w2_e = w2[e].to(torch.float32)
        w2_s = (
            w2_scale[e]
            .to(torch.float32)
            .repeat_interleave(GROUP_N, dim=0)
            .repeat_interleave(GROUP_K, dim=1)
        )
        result = hq_deq[offs] @ (w2_e * w2_s).t()
        out_rows[offs] = (result * w_flat[offs][:, None]).to(output.dtype)
    output.copy_(out_rows.view(num_tokens, top_k, H).sum(dim=1).to(output.dtype))


def _baseline_accuracy_check(actual: object, expected: object) -> None:
    """Accuracy check that compares only ``output``.

    The workspace caches hold the bf16 gate_up and weighted GEMM2 scratch (and
    the silu+quant op mints its own outputs), which the reference does not
    reproduce byte-identically; the GEMM sites are validated end to end by the
    whole-output comparison.
    """
    if isinstance(actual, tuple | list):
        assert isinstance(expected, tuple | list)
        assert_close_with_mismatch_tolerance(
            actual[0], expected[0], atol=1e-1, rtol=1e-1
        )
    elif actual is not None:
        assert_close_with_mismatch_tolerance(actual, expected, atol=1e-1, rtol=1e-1)


# Quantized GEMM kernels can have relatively large numerical differences from
# the baseline, so the accuracy check runs at a loose 1e-1 tolerance. It also
# compares only ``output``: the workspace caches hold gate_up / GEMM2 scratch
# the reference does not reproduce, so autotune_baseline_accuracy_check_fn
# excludes them (see _baseline_accuracy_check).
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
def mega_moe_block_mm(
    run,
    output: torch.Tensor,  # [num_tokens, H] (reduced layer output)
    hidden_states: torch.Tensor,  # [num_tokens, K] fp8
    w1: torch.Tensor,  # [E, N, K] fp8 (gate_up, N = 2 * intermediate)
    w2: torch.Tensor,  # [E, H, intermediate] fp8 (down projection)
    w1_scale: torch.Tensor,  # [E, N // 128, K // 128]
    w2_scale: torch.Tensor,  # [E, H // 128, intermediate // 128]
    a1q_scale: torch.Tensor,  # [num_tokens, K // 128]
    topk_weights: torch.Tensor,  # [num_tokens, top_k]
    topk_ids: torch.Tensor,  # [num_tokens, top_k]
    expert_map: torch.Tensor | None,
    global_num_experts: int,
    workspace13: torch.Tensor,
    workspace2: torch.Tensor,
    clamp_limit: float | None = None,
) -> None:
    unpermuted = run.tunable("unpermuted", BooleanFragment())
    # unpermuted = False
    num_tokens = hidden_states.shape[0]
    top_k = topk_ids.shape[1]
    N = w1.shape[1]
    H = w2.shape[1]
    routed = num_tokens * top_k

    align_block_size = run.tunable(
        "align_block_size",
        PowerOfTwoFragment(
            1,
            helion.next_power_of_2(num_tokens),
            default_val=helion.next_power_of_2(num_tokens),
        ),
    )

    # Fold the top_k combine into GEMM2's store: each weighted routed row
    # atomically accumulates into its token's output row (the child pre-zeroes
    # the output), replacing the mm2_out staging cache + moe_sum.
    fuse_moe_sum = run.tunable("fuse_moe_sum", BooleanFragment())

    # The child GEMM takes split-K as an input (one kernel serves both stages),
    # so each stage gets its own root tunable; their K ranges differ (the
    # hidden dim for the gate_up projection, the intermediate for the down
    # projection). fuse_act_quant folds silu+block-quant into GEMM1's
    # epilogue, which needs the whole K accumulated there.
    intermediate = N // 2
    num_group_i = intermediate // GROUP_K
    split_k_1 = run.tunable(
        "split_k_1",
        PowerOfTwoFragment(
            1, helion.next_power_of_2(hidden_states.shape[1] // GROUP_K)
        ),
    )
    split_k_2 = run.tunable(
        "split_k_2",
        PowerOfTwoFragment(1, helion.next_power_of_2(num_group_i)),
    )
    fuse_act_quant = run.tunable("fuse_act_quant", BooleanFragment())
    fused_act_quant = fuse_act_quant and split_k_1 == 1

    # The permuted mode routes through the standalone moe_route kernel,
    # called as its registered custom op (torch.ops.vllm_helion.moe_route) so
    # it picks its own pre-tuned config and stays out of the composite's
    # joint config space: one fused launch produces the aligned layout
    # (sorted_token_ids / expert_ids), the expert-major permuted activations
    # and scales, inv_perm, and the padded routed-slot count -- replacing
    # _align_routing + moe_permute. Both GEMM stages then gather through
    # inv_perm (GEMM1 also stores permuted, keeping the intermediate
    # contiguous per expert, while GEMM2 scatters its weighted output to the
    # routed slots' token-order rows, so the combine is always a contiguous
    # moe_sum). The unpermuted mode keeps _align_routing's layout:
    # moe_align_block_size wins the alignment-only case, degenerating to the
    # dense layout (no align kernel) at align_block_size == 1. Either way the
    # count flows to both GEMMs as a device tensor -- their grids stay at the
    # worst-case bound and prune the tail in-kernel, so nothing reads back to
    # the host.
    aq = hidden_states
    aq_scale = a1q_scale
    inv_perm = None
    if not unpermuted:
        max_padded, num_blocks = _align_sizes(
            num_tokens, top_k, global_num_experts, align_block_size
        )
        device = hidden_states.device
        sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device=device)
        expert_ids = torch.empty(num_blocks, dtype=torch.int32, device=device)
        permuted_aq = torch.empty(
            (routed, hidden_states.shape[1]), dtype=hidden_states.dtype, device=device
        )
        permuted_scale = torch.empty(
            (routed, a1q_scale.shape[1]), dtype=a1q_scale.dtype, device=device
        )
        inv_perm = torch.empty(routed, dtype=torch.int32, device=device)
        num_tokens_post_pad = torch.empty(1, dtype=torch.int32, device=device)
        torch.ops.vllm_helion.moe_route(
            topk_ids,
            expert_map,
            sorted_token_ids,
            expert_ids,
            num_tokens,
            top_k,
            align_block_size,
            global_num_experts,
            hidden_states,
            a1q_scale,
            permuted_aq,
            permuted_scale,
            inv_perm,
            num_tokens_post_pad,
        )
        aq, aq_scale = permuted_aq, permuted_scale
    else:
        sorted_token_ids, expert_ids, num_tokens_post_pad = _align_routing(
            topk_ids, expert_map, global_num_experts, align_block_size
        )

    # GEMM1: gate_up projection, staged in workspace2 like the Triton baseline
    # (TritonExperts' intermediate_cache1). The fused epilogue writes fp8
    # activations plus per-block scales instead of bf16 rows, both carved out
    # of that cache so the timed region allocates nothing: the fp8 rows first,
    # then the float32 scales (whose offset stays 4-byte aligned because the
    # intermediate is a multiple of GROUP_K).
    cached = _resize_cache(workspace2, (num_tokens, top_k, N))
    if fused_act_quant:
        buf = cached.view(-1).view(torch.uint8)
        n_q = routed * intermediate
        n_s = routed * num_group_i * 4
        hq = buf[:n_q].view(torch.float8_e4m3fn).view(routed, intermediate)
        hq_scale = buf[n_q : n_q + n_s].view(torch.float32).view(routed, num_group_i)
        run(
            moe_block_mm,
            hq,
            aq,
            aq_scale,
            w1,
            w1_scale,
            sorted_token_ids,
            expert_ids,
            num_tokens,
            top_k,
            align_block_size,
            split_k_1,
            num_tokens_post_pad,
            None,
            inv_perm,
            False,  # fuse_moe_sum: the gate_up projection never combines
            clamp_limit,
            hq_scale,
        )
    else:
        gate_up = cached.view(routed, N)
        run(
            moe_block_mm,
            gate_up,
            aq,
            aq_scale,
            w1,
            w1_scale,
            sorted_token_ids,
            expert_ids,
            num_tokens,
            top_k,
            align_block_size,
            split_k_1,
            num_tokens_post_pad,
            None,
            inv_perm,
            False,  # fuse_moe_sum: the gate_up projection never combines
            None,  # clamp_limit: applied before quantizing, not in the GEMM
            None,  # output_scale: no fused epilogue
        )

        # silu_and_mul + per-block fp8 quant at the full routed M. With a clamp
        # limit this runs as two steps and needs a bf16 activation buffer,
        # staged in the still-unused workspace13 cache.
        act = None
        if clamp_limit is not None:
            act = _resize_cache(workspace13, (num_tokens, top_k, intermediate)).view(
                routed, intermediate
            )
        hq, hq_scale = _silu_mul_block_quant(gate_up, clamp_limit, act)

    # GEMM2: down projection over the per-routed-row quantized intermediate.
    # Each row is scaled by its routed weight in the epilogue (Triton's
    # mul_routed_weight). Unfused, the weighted rows are staged in workspace13
    # (TritonExperts' intermediate_cache3 role) and reduced over top_k by
    # moe_sum; fused (fuse_moe_sum), GEMM2 stores straight into the layer
    # output, accumulating each token's top_k contributions atomically.
    mm2_out = (
        None
        if fuse_moe_sum
        else _resize_cache(workspace13, (num_tokens, top_k, H)).view(routed, H)
    )
    run(
        moe_block_mm,
        output if fuse_moe_sum else mm2_out,
        hq,
        hq_scale,
        w2,
        w2_scale,
        sorted_token_ids,
        expert_ids,
        num_tokens,
        top_k,
        align_block_size,
        split_k_2,
        num_tokens_post_pad,
        topk_weights,
        inv_perm,
        fuse_moe_sum,
    )

    # Combine: reduce the weighted routed rows over top_k into the layer
    # output. GEMM2 writes token-order rows in both modes, so this is always
    # a contiguous moe_sum; the fused combine skipped it above. The routed ids
    # make it the pad-aware variant, which skips slots routed to another rank:
    # both layouts are compact, so GEMM2 never writes those rows and their
    # contribution from this rank is zero by construction.
    if not fuse_moe_sum:
        ops.moe_sum(mm2_out.view(num_tokens, top_k, H), output, topk_ids, expert_map)
