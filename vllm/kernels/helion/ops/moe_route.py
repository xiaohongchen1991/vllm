# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helion fused routing kernel: moe_align_block_size's layout and moe_permute's
expert-major activation rows from one expert census.

Called by the mega composite's permuted branch (through its registered custom
op, with configs tuned separately from the composite's joint space); kept
standalone for testing and optimization. Four sequential
tile passes (helion inserts the cross-pass dependencies):

0. Zero the census scratch (separate from pass 1: the histogram atomics must
   not race the zeroing).
1. Fill the aligned-layout buffers (padding marker / -1 tail) and histogram
   each slot's (locally mapped) expert.
2. Exclusive per-expert offsets from the census -- each program computes its
   chunk's offsets with a bounded masked sum, so the tiles stay small for any
   expert count.
3. Place every slot: gather its expert's block offset, claim a per-expert
   rank with one atomic, and scatter to its aligned block position. When the
   permutation source is passed, also compute its expert-major row
   (row offset + rank), store ``inv_perm``, and copy the token's activation
   and scale rows through.

The permutation is keyed on ``source_aq`` being non-None (None checks resolve
at trace time; booleans do not). Atomic ranks make the within-expert order
non-deterministic -- any slot -> row bijection is a valid routing, and every
buffer value is computed by exactly one program, so the maps are correct
either way.

Interface notes: ``sorted_token_ids`` / ``expert_ids`` are caller-allocated
with the worst-case ``_align_sizes`` shapes (static per specialization);
``num_tokens_post_padded`` (optional, [1] int32) receives the actual padded
routed-slot count (total blocks x m_block_size) so consumers can prune their
worst-case grids in-kernel without a D2H sync; the
census scratch (counts / ranks / row offsets / block offsets) is allocated
inside the kernel and never escapes it. Off-rank slots (expert_map -> -1)
never write: their blocks do not exist (the expert_ids -1 tail covers them)
and their stores are masked.

The tuned cases are permuted-only (the mega's sole use of this kernel) at
``m_block_size = next_power_of_two(num_tokens)`` -- the kernel requires a
power-of-two block size (its pass-1 lane arange), and the runtime
``m_block_size`` pads the layout but is not a tuning dimension -- and mirror
the mega's per-model (E, EP, K)
deployments with a linear rank-0 ``expert_map`` under EP.
"""

import math
from itertools import product
from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
import helion.language as hl

from vllm.kernels.helion.register import register_kernel


def _align_sizes(
    num_tokens: int, top_k: int, E: int, m_block_size: int
) -> tuple[int, int]:
    numel = num_tokens * top_k
    max_padded = numel + E * (m_block_size - 1)
    if numel < E:
        max_padded = min(numel * m_block_size, max_padded)
    num_blocks = (max_padded + m_block_size - 1) // m_block_size
    return max_padded, num_blocks


def _simulate_topk_ids(num_tokens: int, E: int, top_k: int) -> torch.Tensor:
    """Per-token top_k expert ids from a simulated uniform router, seeded per
    shape for reproducibility."""
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
    top_k: int,
    num_tokens: int,
    fp8: torch.dtype,
    ep_size: int = 1,
) -> tuple[Any, ...]:
    assert E % ep_size == 0
    num_local_experts = E // ep_size

    # The mega's permuted branch is the only caller, and its tuning pairs
    # align_block_size with num_tokens: keep that pairing here, rounded up to
    # a power of two (the kernel's pass-1 lane arange requires it).
    m_block_size = helion.next_power_of_2(num_tokens)

    scale_dtype = torch.float32
    routed = num_tokens * top_k
    num_group_k = K // 128
    max_padded, num_blocks = _align_sizes(num_tokens, top_k, E, m_block_size)

    topk_ids = _simulate_topk_ids(num_tokens, E, top_k)
    sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(num_blocks, dtype=torch.int32, device="cuda")

    scale = 1.0 / math.sqrt(K)
    source_aq = (scale * (0.5 + torch.rand(num_tokens, K, device="cuda"))).to(fp8)
    # Transposed scale layout, as the mega's ue8m0 activation scales arrive:
    # [num_tokens, K // 128] as a .t() view.
    source_scale = (0.5 + torch.rand(num_group_k, num_tokens, device="cuda")).t()
    permuted_aq = torch.empty(routed, K, dtype=fp8, device="cuda")
    permuted_scale = torch.empty(routed, num_group_k, dtype=scale_dtype, device="cuda")
    inv_perm = torch.empty(routed, dtype=torch.int32, device="cuda")
    num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device="cuda")

    # A TP+EP rank's view (linear placement, this rank first): on-rank experts
    # reindexed from 0, off-rank -1, as the mega's permuted branch passes it.
    if ep_size == 1:
        expert_map = None
    else:
        expert_map = torch.full((E,), -1, dtype=torch.int32, device="cuda")
        expert_map[:num_local_experts] = torch.arange(
            num_local_experts, dtype=torch.int32, device="cuda"
        )

    return (
        topk_ids,
        expert_map,
        sorted_token_ids,
        expert_ids,
        num_tokens,
        top_k,
        m_block_size,
        E,
        source_aq,
        source_scale,
        permuted_aq,
        permuted_scale,
        inv_perm,
        num_tokens_post_padded,
    )


def generate_inputs() -> dict[CaseKey, tuple[Any, ...]]:
    fp8: torch.dtype = torch.float8_e4m3fn

    # (E, top_k, EP, K) entries mirroring mega_moe_block_mm.py's
    # e_topk_ep_kn_list (N is the GEMM width -- irrelevant to routing). Every
    # case is permuted (the mega's sole use) and runs at m_block_size =
    # num_tokens, which is not a tuning dimension.
    # - Qwen3-30B-A3B, 1x H100, TP=1/EP=1: 128 experts, hidden 2048.
    # - Qwen3-235B-A22B, 4x H100, TP=4/EP=4: 128 experts, hidden 4096.
    # - GLM-5.3, 8x H200, TP=8/EP=8: 256 experts, hidden 6144.
    e_topk_ep_k_list: list[tuple[int, int, int, int]] = [
        (128, 8, 1, 2048),  # Qwen3-30B-A3B
        (128, 8, 4, 4096),  # Qwen3-235B-A22B
        (256, 8, 8, 6144),  # GLM-5.3
    ]
    num_tokens_list = [4, 8, 16, 32, 64]

    inputs = {}
    for num_tokens, (E, top_k, ep_size, K) in product(
        num_tokens_list, e_topk_ep_k_list
    ):
        inputs[
            CaseKey(
                {
                    "E": E,
                    "K": K,
                    "topk": top_k,
                    "num_tokens": num_tokens,
                    "permuted": True,
                }
            )
        ] = _generate_input(E, K, top_k, num_tokens, fp8, ep_size)

    return inputs


_pick_cache: dict[tuple[int, int, int, int, bool], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config: exact (E, K, top_k, permuted)
    preferred, num_tokens bucketed to the smallest tuned value >= runtime.
    m_block_size is not matched -- cases are tuned at m_block_size =
    next_power_of_two(num_tokens), and the runtime value only pads the
    layout."""

    if not config_keys:
        return None

    topk_ids = args[0]
    num_tokens, top_k = topk_ids.shape
    num_experts = args[7]
    source_aq = args[8]
    permuted = source_aq is not None
    K = source_aq.shape[1] if permuted else 0

    cache_key = (num_experts, K, top_k, num_tokens, permuted)
    if cache_key in _pick_cache:
        return _pick_cache[cache_key]

    candidates: list[CaseKey] = []
    for key in config_keys:
        if key.is_default():
            continue
        if not all(k in key for k in ("E", "K", "topk", "num_tokens", "permuted")):
            continue
        candidates.append(key)

    if not candidates:
        _pick_cache[cache_key] = None
        return None

    def sort_key(key: CaseKey) -> tuple[int, int, int, int, int, int]:
        nt = key["num_tokens"]
        covers = 0 if nt >= num_tokens else 1
        return (
            key["permuted"] != permuted,
            abs(key["topk"] - top_k),
            abs(key["E"] - num_experts),
            abs(key["K"] - K),
            covers,
            abs(nt - num_tokens),
        )

    result = min(candidates, key=sort_key)
    _pick_cache[cache_key] = result
    return result


def fake_impl(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens: int,
    top_k: int,
    m_block_size: int,
    num_experts: int,
    source_aq: torch.Tensor | None = None,
    source_scale: torch.Tensor | None = None,
    permuted_aq: torch.Tensor | None = None,
    permuted_scale: torch.Tensor | None = None,
    inv_perm: torch.Tensor | None = None,
    num_tokens_post_padded: torch.Tensor | None = None,
) -> None:
    return


def baseline(
    topk_ids: torch.Tensor,
    expert_map: torch.Tensor | None,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens: int,
    top_k: int,
    m_block_size: int,
    num_experts: int,
    source_aq: torch.Tensor | None = None,
    source_scale: torch.Tensor | None = None,
    permuted_aq: torch.Tensor | None = None,
    permuted_scale: torch.Tensor | None = None,
    inv_perm: torch.Tensor | None = None,
    num_tokens_post_padded: torch.Tensor | None = None,
) -> None:
    """Eager torch reference (the mega's routing without torch.compile): a
    stable argsort gives the expert-major slot order; searchsorted over the
    sorted keys gives the per-expert offsets."""
    device = topk_ids.device
    routed = num_tokens * top_k

    e = topk_ids.reshape(-1).to(torch.int64)
    if expert_map is not None:
        e = expert_map[e].to(torch.int64)
    valid = e >= 0
    # Off-rank slots key to a trash bin past the last expert, so they sort to
    # the end and never disturb the valid experts' offsets.
    keyed = torch.where(valid, e, torch.full_like(e, num_experts))

    order = torch.argsort(keyed, stable=True)
    experts_sorted = keyed[order]
    pos = torch.arange(routed, device=device)
    # boundaries[expert] = first sorted position of that expert (exclusive
    # offset); boundaries[num_experts] = number of valid slots.
    boundaries = torch.searchsorted(
        experts_sorted, torch.arange(num_experts + 1, device=device)
    )
    counts = (boundaries[1:] - boundaries[:-1]).to(torch.int32)
    blocks_per_expert = (counts.to(torch.int64) + (m_block_size - 1)) // m_block_size
    block_offs = torch.cumsum(blocks_per_expert, 0) - blocks_per_expert
    if num_tokens_post_padded is not None:
        num_tokens_post_padded.copy_(
            (blocks_per_expert.sum() * m_block_size).to(torch.int32).reshape(1)
        )

    # Aligned layout: padding marker / -1 tail, then the real blocks/slots.
    sorted_token_ids.fill_(routed)
    expert_ids.fill_(-1)
    real = experts_sorted < num_experts
    e_safe = torch.where(real, experts_sorted, 0)
    rank = pos - boundaries[e_safe]
    blk = block_offs[e_safe] + rank // m_block_size
    slot = blk * m_block_size + rank % m_block_size
    sorted_token_ids[slot[real]] = order[real].to(torch.int32)
    expert_ids[blk[real]] = experts_sorted[real].to(torch.int32)

    if source_aq is not None:
        assert source_scale is not None
        assert permuted_aq is not None
        assert permuted_scale is not None
        assert inv_perm is not None
        # The permuted layout is the sort order itself: row i holds slot
        # order[i]; inv_perm maps each slot back to its row.
        rows = order // top_k
        permuted_aq.copy_(source_aq[rows])
        permuted_scale.copy_(source_scale[rows])
        inv_perm[order[real]] = pos[real].to(torch.int32)


def _baseline_accuracy_check(actual: object, expected: object) -> None:
    """Accuracy check comparing the routing maps semantically.

    The census scratch is kernel-internal, so only the layouts it feeds are
    observable: expert_ids must match byte-for-byte, while the within-expert
    slot order is not deterministic (the kernel claims ranks with atomics,
    the reference with a stable sort), so sorted_token_ids is compared as
    per-block slot sets and the permutation as: gathering each valid slot's
    row through inv_perm reproduces its token's source row.
    """
    if not isinstance(actual, tuple | list):
        assert actual is None
        return
    assert isinstance(expected, tuple | list)
    a, ref = list(actual), list(expected)
    # sorted(2) expert_ids(3) num_tokens(4) top_k(5) m_block_size(6) E(7)
    # src_aq(8) src_scale(9) permuted_aq(10) permuted_scale(11) inv_perm(12)
    # ntp(13)
    assert torch.equal(a[3], ref[3])
    if a[13] is not None:
        assert torch.equal(a[13], ref[13])

    topk_ids = a[0]
    num_tokens, top_k = topk_ids.shape
    m_block_size = a[6]
    routed = num_tokens * top_k
    sa, sr, ea = a[2], ref[2], a[3]
    for blk in range(ea.numel()):
        if int(ea[blk]) < 0:
            continue
        got = sa[blk * m_block_size : (blk + 1) * m_block_size]
        got = got[got < routed].sort().values
        want = sr[blk * m_block_size : (blk + 1) * m_block_size]
        want = want[want < routed].sort().values
        assert torch.equal(got, want)

    if a[8] is not None:
        source_aq, source_scale = a[8], a[9]
        inv = a[12].to(torch.long)
        slots = torch.arange(routed, device=source_aq.device)
        topk_flat = topk_ids.reshape(-1)
        # Under EP the global ids are all non-negative; validity is whether
        # the slot's expert is on this rank.
        if a[1] is not None:
            valid_slots = a[1][topk_flat.long()] >= 0
        else:
            valid_slots = topk_flat >= 0
        # inv_perm must land each valid slot on a row holding its token's
        # source activation and scale (exact byte copies). Off-rank slots'
        # inv_perm entries are never written, so clamp their garbage before
        # the gather (an unmasked OOB index faults the index kernel).
        inv = torch.where(valid_slots, inv, torch.zeros_like(inv))
        assert torch.equal(
            a[10][inv][valid_slots], source_aq[slots // top_k][valid_slots]
        )
        assert torch.equal(
            a[11][inv][valid_slots], source_scale[slots // top_k][valid_slots]
        )


# The routing maps are integer metadata (no fp8 numerics), so the check runs
# at exact equality; the non-deterministic within-expert order is handled
# semantically (see _baseline_accuracy_check).
@register_kernel(
    mutates_args=[
        "sorted_token_ids",
        "expert_ids",
        "permuted_aq",
        "permuted_scale",
        "inv_perm",
        "num_tokens_post_padded",
    ],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        autotune_baseline_fn=baseline,
        autotune_baseline_accuracy_check_fn=_baseline_accuracy_check,
        ignore_warnings=[
            helion.exc.TensorOperationInWrapper,
            helion.exc.ProcessGroupNameNotFound,
        ],
    ),
)
def moe_route(
    topk_ids: torch.Tensor,  # [num_tokens, top_k] int32
    expert_map: torch.Tensor
    | None,  # [global_num_experts] global -> local, -1 = off-rank
    sorted_token_ids: torch.Tensor,  # [max_padded] int32 (out)
    expert_ids: torch.Tensor,  # [num_blocks] int32 (out)
    num_tokens: int,
    top_k: int,
    m_block_size: int,
    num_experts: int,
    source_aq: torch.Tensor
    | None = None,  # [num_tokens, K] (in; None = skip permutation)
    source_scale: torch.Tensor | None = None,  # [num_tokens, K // 128] (in)
    permuted_aq: torch.Tensor | None = None,  # [routed, K] (out)
    permuted_scale: torch.Tensor | None = None,  # [routed, K // 128] (out)
    inv_perm: torch.Tensor | None = None,  # [routed] int32 (out)
    num_tokens_post_padded: torch.Tensor
    | None = None,  # [1] int32 (out: padded routed-slot count)
) -> None:
    routed = num_tokens * top_k

    hl.specialize(num_tokens)
    hl.specialize(top_k)
    hl.specialize(m_block_size)
    hl.specialize(num_experts)

    topk_flat = topk_ids.reshape(-1)
    assert topk_flat.numel() == routed
    assert topk_ids.dtype == torch.int32
    assert expert_ids.dtype == torch.int32
    # Pass 1's lane count (tile block size x m_block_size) feeds tl.arange,
    # which requires a power-of-two length -- so the block size must be one.
    assert m_block_size > 0 and (m_block_size & (m_block_size - 1)) == 0

    # Census scratch, allocated fresh per launch: counts / ranks are zeroed
    # by pass 0, and pass 2 fully writes the offsets.
    counts_buf = torch.empty(num_experts, dtype=torch.int32, device=topk_ids.device)
    ranks_buf = torch.empty(num_experts, dtype=torch.int32, device=topk_ids.device)
    offs_buf = torch.empty(num_experts, dtype=torch.int32, device=topk_ids.device)
    block_offs_buf = torch.empty(num_experts, dtype=torch.int32, device=topk_ids.device)

    # Worst-case aligned-layout sizes (the _align_sizes formulas): the
    # scatter writes only real blocks/slots, so the fills carry the padding
    # marker / -1 tail.
    max_padded = routed + num_experts * (m_block_size - 1)
    if routed < num_experts:
        max_padded = min(routed * m_block_size, max_padded)
    num_blocks = (max_padded + m_block_size - 1) // m_block_size
    assert sorted_token_ids.numel() == max_padded
    assert expert_ids.numel() == num_blocks

    num_group_k = 0
    if source_aq is not None:
        assert source_scale is not None
        assert permuted_aq is not None
        assert permuted_scale is not None
        assert inv_perm is not None
        K = source_aq.shape[1]
        hl.specialize(K)
        num_group_k = K // 128
        assert permuted_aq.shape == (routed, K)
        assert permuted_scale.shape == (routed, num_group_k)
        assert inv_perm.numel() == routed
        assert source_scale.ndim == 2

    # Pass 0: zero the census scratch. Separate from pass 1 -- the histogram
    # atomics must not race the zeroing.
    for tile_e in hl.tile(num_experts):
        counts_buf[tile_e] = hl.zeros([tile_e], dtype=counts_buf.dtype)
        ranks_buf[tile_e] = hl.zeros([tile_e], dtype=ranks_buf.dtype)

    # Pass 1: fills + histogram.
    for tile_b in hl.tile(num_blocks):
        idx = tile_b.begin * m_block_size + hl.arange(tile_b.block_size * m_block_size)
        hl.store(
            sorted_token_ids,
            [idx],
            hl.full(
                [tile_b.block_size * m_block_size], routed, dtype=sorted_token_ids.dtype
            ),
            extra_mask=idx < max_padded,
        )
        expert_ids[tile_b] = hl.full([tile_b], -1, dtype=expert_ids.dtype)
        e = hl.load(topk_flat, [idx], extra_mask=idx < routed)
        if expert_map is not None:
            e = hl.load(expert_map, [torch.where(e >= 0, e, 0)], extra_mask=e >= 0).to(
                torch.int32
            )

        valid = (idx < routed) & (e >= 0)
        e_safe = torch.where(valid, e, 0)
        hl.atomic_add(counts_buf, [e_safe], torch.where(valid, 1, 0))

    # Pass 2: exclusive per-expert offsets. Each program computes its chunk's
    # offsets from the full census with a bounded masked sum (no scan carry,
    # no per-slot rescan).
    for tile_e in hl.tile(num_experts):
        e_all = hl.arange(num_experts)
        e_idx = tile_e.begin + hl.arange(tile_e.block_size)
        counts = counts_buf[e_all]
        if source_aq is not None:
            offs_buf[tile_e] = torch.where(
                e_all[None, :] < e_idx[:, None], counts[None, :], 0
            ).sum(dim=1)

        blocks_per_expert = (counts + (m_block_size - 1)) // m_block_size
        block_offs_buf[tile_e] = torch.where(
            e_all[None, :] < e_idx[:, None], blocks_per_expert[None, :], 0
        ).sum(dim=1)

        # The actual padded routed-slot count (total blocks x block size) so
        # downstream consumers can prune their worst-case grids in-kernel.
        # Every program reads the full census (same value), so only the first
        # computes and stores it -- a single writer, no redundant reductions.
        # Only real blocks exist, so the count is a multiple of m_block_size
        # by construction.
        if num_tokens_post_padded is not None and tile_e.begin == 0:
            total_blocks = blocks_per_expert[None, :].sum(dim=1)
            hl.store(
                num_tokens_post_padded,
                [hl.arange(1)],
                total_blocks * m_block_size,
            )

    # Pass 3: place every slot -- gather its expert's block offset, claim a
    # per-expert rank with one atomic, and scatter. The slot block is fixed
    # so the row-copy tiles stay small at untuned defaults.
    for tile_r in hl.tile(routed):
        e = topk_flat[tile_r]
        if expert_map is not None:
            e = expert_map[e].to(torch.int32)
        valid = e >= 0
        e_safe = torch.where(valid, e, 0)

        rank = hl.atomic_add(ranks_buf, [e_safe], torch.where(valid, 1, 0))

        blk = hl.load(block_offs_buf, [e_safe]) + rank // m_block_size
        hl.store(
            sorted_token_ids,
            [blk * m_block_size + rank % m_block_size],
            tile_r.index,
            extra_mask=valid,
        )
        hl.store(expert_ids, [blk], e, extra_mask=valid)
        if source_aq is not None:
            perm_row = offs_buf[e_safe] + rank

            hl.store(inv_perm, [tile_r], perm_row, extra_mask=valid)
            src = torch.where(valid, tile_r.index, 0) // top_k
            dst = torch.where(valid, perm_row, 0)
            for tile_k in hl.tile(K):
                a = source_aq[src, tile_k]
                hl.store(permuted_aq, [dst, tile_k], a, extra_mask=valid[:, None])
            for tile_g in hl.tile(num_group_k):
                sc = source_scale[src, tile_g]
                hl.store(permuted_scale, [dst, tile_g], sc, extra_mask=valid[:, None])
