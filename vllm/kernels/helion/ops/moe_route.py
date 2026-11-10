# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helion fused routing kernel: moe_align_block_size's layout and moe_permute's
expert-major activation rows from one expert census.

Called by the mega composite's permuted branch (through its registered custom
op, with configs tuned separately from the composite's joint space); kept
standalone for testing and optimization. Four sequential
tile passes (helion inserts the cross-pass dependencies):

0. Zero the census scratch (separate from pass 1: the census atomics must not
   race the zeroing).
1. Census each slot's (locally mapped) expert, claiming its within-expert rank
   in the same atomic -- its return value -- and store that rank per slot.
2. Exclusive per-expert offsets from the census: four programs (``hl.grid(4)``)
   each own one output -- the permutation row offsets, the block offsets, the
   padded routed-slot count, and the exact on-rank slot count -- so each
   computes just its own from one census load instead of one program issuing
   all four back to back.
3. Place every slot: gather its expert's block offset and the rank pass 1
   claimed, scatter to its aligned block position, and sentinel-fill the
   expert's padding tail (the only padding a consumer can read). When the
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
``num_tokens_post_padded`` (optional, [1] int32) receives the padded routed-slot
count (total blocks x m_block_size) and ``num_valid`` (optional, [1] int32) the
exact on-rank slot count, so each consumer prunes its worst-case grid in-kernel
without a D2H sync -- in *its own* row space: the GEMMs iterate aligned blocks,
whose last block can start past the on-rank slots' row range, while the act-quant
stage iterates compact rows; the
census scratch (counts / ranks / row offsets / block offsets) is allocated
inside the kernel and never escapes it. Off-rank slots (expert_map -> -1)
never write: their blocks do not exist at all, so no block, slot or padding
entry is emitted for them and every access is masked. ``num_experts`` is the
global count and ``num_local_experts`` this rank's share, so the picker can key
its case on
EP = E / local: ``expert_map``'s gathers are trace-time conditional, so the
two EP buckets are different kernels whose per-op config lists differ in
length.

Each tuned case carries several inputs, one per ``m_block_size`` the mega's
``align_block_size`` tunable can hand it (the small-batch formula and the token
count): the runtime block size pads the layout but is not itself a tuning
dimension, so one config is tuned across all of them rather than fitted to the
default. Both values are powers of two, mirroring that tunable's fragment. The
cases mirror the mega's per-model (E, EP, K) deployments with a linear rank-0
``expert_map`` under EP.
"""

import math
from itertools import product
from typing import Any

import torch

from vllm.kernels.helion.case_key import CaseKey
from vllm.kernels.helion.inputs import CaseInputs
from vllm.utils.import_utils import has_helion

if not has_helion():
    raise ImportError(
        "Helion kernel requires helion to be installed. "
        "Install it with: pip install helion"
    )

import helion
import helion.language as hl

from vllm.kernels.helion.register import register_kernel

# Masked atomic add on a 1-D index: lanes that must not participate (padded
# slots, off-rank experts) are skipped instead of being redirected to a live
# location, which would serialize every such lane on that address. Helion's
# ``hl.atomic_add`` takes no mask, so the Triton atomic is emitted directly; the
# expression's value is the location's previous content.
#
# The barrier after the phase that uses it is required, not cosmetic: helion's
# dependency analysis cannot see an opaque snippet's memory effects, so without
# it the consumer phase is not ordered after the producer.
_MASKED_ATOMIC_ADD = "tl.atomic_add({p} + {i}, 1, mask={m}, sem='relaxed')"


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


def _m_block_size_candidates(num_tokens: int, top_k: int, E: int) -> list[int]:
    """The ``m_block_size`` values a case is tuned across.

    Endpoints only. ``m_block_size`` rescales the padded routed-slot count the
    GEMMs iterate (``num_tokens_post_padded``) without changing the kernel's
    structure, so a config measured at both ends
    bounds everything the mega's ``align_block_size`` tunable can pick, and an
    interior point would only interpolate between them at a third of the search
    time. Duplicates collapse; both values are powers of two, like that
    tunable's fragment.
    """
    # lo = max(2, helion.next_power_of_2(num_tokens * top_k // E))
    hi = helion.next_power_of_2(num_tokens)
    return sorted({hi})


def _generate_input(
    E: int,
    K: int,
    top_k: int,
    num_tokens: int,
    fp8: torch.dtype,
    ep_size: int = 1,
    permuted: bool = True,
    m_block_size: int | None = None,
) -> tuple[Any, ...]:
    assert E % ep_size == 0
    num_local_experts = E // ep_size

    # ``m_block_size`` only pads the layout; the default is the small-batch
    # formula, the low end of what generate_inputs tunes over.
    if m_block_size is None:
        m_block_size = max(2, helion.next_power_of_2(num_tokens * top_k // E))

    scale_dtype = torch.float32
    routed = num_tokens * top_k
    num_group_k = K // 128
    max_padded, num_blocks = _align_sizes(num_tokens, top_k, E, m_block_size)

    topk_ids = _simulate_topk_ids(num_tokens, E, top_k)
    sorted_token_ids = torch.empty(max_padded, dtype=torch.int32, device="cuda")
    expert_ids = torch.empty(num_blocks, dtype=torch.int32, device="cuda")

    if permuted:
        scale = 1.0 / math.sqrt(K)
        source_aq = (scale * (0.5 + torch.rand(num_tokens, K, device="cuda"))).to(fp8)
        # Transposed scale layout, as the mega's ue8m0 activation scales
        # arrive: [num_tokens, K // 128] as a .t() view.
        source_scale = (0.5 + torch.rand(num_group_k, num_tokens, device="cuda")).t()
        permuted_aq = torch.empty(routed, K, dtype=fp8, device="cuda")
        permuted_scale = torch.empty(
            routed, num_group_k, dtype=scale_dtype, device="cuda"
        )
        inv_perm = torch.empty(routed, dtype=torch.int32, device="cuda")
    else:
        # Routing-only view: the kernel skips the row copies and the permutation.
        source_aq = source_scale = permuted_aq = permuted_scale = inv_perm = None
    num_tokens_post_padded = torch.empty(1, dtype=torch.int32, device="cuda")
    num_valid = torch.empty(1, dtype=torch.int32, device="cuda")

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
        num_local_experts,
        source_aq,
        source_scale,
        permuted_aq,
        permuted_scale,
        inv_perm,
        num_tokens_post_padded,
        num_valid,
    )


def generate_inputs() -> dict[CaseKey, CaseInputs]:
    fp8: torch.dtype = torch.float8_e4m3fn

    # Hardcoded switch: the mega's permuted branch is the only production
    # caller, so cases are generated permuted. Set this to False to exercise the
    # routing-only view (no permuted activations/scales, no inv_perm) and tune
    # or check that path.
    permuted = True

    # (E, top_k, EP, K) entries mirroring mega_moe_block_mm.py's
    # e_topk_ep_kn_list (N is the GEMM width -- irrelevant to routing).
    # - Qwen3-30B-A3B, 1x H100, TP=1/EP=1: 128 experts, hidden 2048.
    # - Qwen3-235B-A22B, 4x H100, TP=4/EP=4: 128 experts, hidden 4096.
    # - GLM-5.3, 8x H200, TP=8/EP=8: 256 experts, hidden 6144.
    e_topk_ep_k_list: list[tuple[int, int, int, int]] = [
        # (128, 8, 1, 2048),  # Qwen3-30B-A3B
        # (128, 8, 4, 4096),  # Qwen3-235B-A22B
        (256, 8, 8, 6144),  # GLM-5.3
        # (256, 8, 1, 6144),  # GLM-5.3
    ]
    num_tokens_list = [32]

    inputs = {}
    for num_tokens, (E, top_k, ep_size, K) in product(
        num_tokens_list, e_topk_ep_k_list
    ):
        # One input per alignment the mega can ask for: the routing draw is
        # identical across them (seeded by shape), so the search scores each
        # candidate across block sizes instead of fitting the default alone.
        inputs[
            CaseKey(
                {
                    "E": E,
                    "K": K,
                    "topk": top_k,
                    "num_tokens": num_tokens,
                    "EP": ep_size,
                    "permuted": permuted,
                }
            )
        ] = tuple(
            _generate_input(E, K, top_k, num_tokens, fp8, ep_size, permuted, mbs)
            for mbs in _m_block_size_candidates(num_tokens, top_k, E)
        )

    return inputs


_pick_cache: dict[tuple[int, int, int, int, int, bool], CaseKey | None] = {}


def pick_config(args: tuple[Any, ...], config_keys: list[CaseKey]) -> CaseKey | None:
    """Pick the best pre-tuned config: exact (E, K, top_k, EP, permuted)
    preferred, num_tokens bucketed to the smallest tuned value >= runtime.
    m_block_size is not matched -- each case is tuned across the block sizes
    the mega can pick (see ``_m_block_size_candidates``), and the runtime value
    only pads the layout."""

    if not config_keys:
        return None

    topk_ids = args[0]
    num_tokens, top_k = topk_ids.shape
    num_experts = args[7]
    num_local_experts = args[8]
    source_aq = args[9]
    permuted = source_aq is not None
    K = source_aq.shape[1] if permuted else 0
    ep_size = num_experts // num_local_experts

    cache_key = (num_experts, ep_size, K, top_k, num_tokens, permuted)
    if cache_key in _pick_cache:
        return _pick_cache[cache_key]

    candidates: list[CaseKey] = []
    for key in config_keys:
        if key.is_default():
            continue
        if not all(
            k in key for k in ("E", "K", "topk", "num_tokens", "EP", "permuted")
        ):
            continue
        if key["permuted"] != permuted:
            # The permuted case adds the two row-copy loops, so its block_sizes
            # (and grid shape) do not transfer across the switch; better to fall
            # back to the default config than to hand over a mismatched one.
            continue
        if key["EP"] != ep_size:
            # EP changes the traced kernel (expert_map's gathers are trace-time
            # conditional), so its per-op indexing / eviction lists have a
            # different arity: a section from another EP bucket would silently
            # shift every entry past the first gather.
            continue
        candidates.append(key)

    if not candidates:
        _pick_cache[cache_key] = None
        return None

    def sort_key(key: CaseKey) -> tuple[int, int, int, int, int]:
        nt = key["num_tokens"]
        covers = 0 if nt >= num_tokens else 1
        return (
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
    num_local_experts: int,
    source_aq: torch.Tensor | None = None,
    source_scale: torch.Tensor | None = None,
    permuted_aq: torch.Tensor | None = None,
    permuted_scale: torch.Tensor | None = None,
    inv_perm: torch.Tensor | None = None,
    num_tokens_post_padded: torch.Tensor | None = None,
    num_valid: torch.Tensor | None = None,
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
    num_local_experts: int,
    source_aq: torch.Tensor | None = None,
    source_scale: torch.Tensor | None = None,
    permuted_aq: torch.Tensor | None = None,
    permuted_scale: torch.Tensor | None = None,
    inv_perm: torch.Tensor | None = None,
    num_tokens_post_padded: torch.Tensor | None = None,
    num_valid: torch.Tensor | None = None,
) -> None:
    """Eager torch reference (the mega's routing without torch.compile).

    The expert-major slot order is a counting sort -- per-key exclusive offsets
    plus each element's rank within its key -- so the reference compiles under
    ``torch.compile``: inductor's Triton backend only lowers ``aten.sort`` inside
    a persistent reduction, which a stable argsort is not.
    """
    device = topk_ids.device
    routed = num_tokens * top_k

    e = topk_ids.reshape(-1).to(torch.int64)
    if expert_map is not None:
        e = expert_map[e].to(torch.int64)
    valid = e >= 0
    # Off-rank slots key to a trash bin past the last expert, so they sort to
    # the end and never disturb the valid experts' offsets.
    keyed = torch.where(valid, e, torch.full_like(e, num_experts))

    # Stable expert-major order as a counting sort: one [keys, routed]
    # comparison matrix gives both the per-key offsets and each element's rank
    # within its key, and the offsets are also the boundaries the layout needs.
    onehot = keyed[None, :] == torch.arange(num_experts + 1, device=device)[:, None]
    rank_in_key = onehot.to(torch.int32).cumsum(1).gather(0, keyed[None, :])[0] - 1
    counts_by_key = onehot.sum(1)
    # boundaries[expert] = first sorted position of that expert (exclusive
    # offset); boundaries[num_experts] = number of valid slots.
    boundaries = counts_by_key.cumsum(0) - counts_by_key
    pos = torch.arange(routed, device=device)
    order = torch.empty(routed, dtype=torch.long, device=device)
    order[boundaries[keyed] + rank_in_key] = pos
    experts_sorted = keyed[order]
    counts = (boundaries[1:] - boundaries[:-1]).to(torch.int32)
    blocks_per_expert = (counts.to(torch.int64) + (m_block_size - 1)) // m_block_size
    block_offs = torch.cumsum(blocks_per_expert, 0) - blocks_per_expert
    if num_tokens_post_padded is not None:
        num_tokens_post_padded.copy_(
            (blocks_per_expert.sum() * m_block_size).to(torch.int32).reshape(1)
        )
    if num_valid is not None:
        num_valid.copy_(boundaries[num_experts].to(torch.int32).reshape(1))

    real = experts_sorted < num_experts
    e_safe = torch.where(real, experts_sorted, 0)
    rank = pos - boundaries[e_safe]
    blk = block_offs[e_safe] + rank // m_block_size
    slot = blk * m_block_size + rank % m_block_size

    # Aligned layout: padding marker / -1 tail, then the real blocks/slots.
    # Real rows land through fixed-shape scatters: boolean-mask assignment needs
    # the number of selected rows on the host, which CUDA-graph capture of this
    # reference forbids. A one-entry staging buffer makes room for that, its
    # last slot absorbing the off-rank lanes.
    def scatter_masked(
        out: torch.Tensor,  # [n], pre-filled with the padding marker
        index: torch.Tensor,  # [routed] destination of every slot
        value: torch.Tensor,  # [routed] value of every slot
        fill: int,  # padding marker / -1 tail
    ) -> None:
        n = out.numel()
        staged = torch.full((n + 1,), fill, dtype=out.dtype, device=device)
        staged.scatter_(
            0,
            torch.where(real, index, n),
            torch.where(real, value, fill).to(out.dtype),
        )
        out.copy_(staged[:n])

    scatter_masked(sorted_token_ids, slot, order, routed)
    scatter_masked(expert_ids, blk, experts_sorted, -1)

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
        # Off-rank slots get 0 rather than an unwritten (uninitialized) entry.
        scatter_masked(inv_perm, order, pos, 0)


def _baseline_accuracy_check(actual: object, expected: object) -> None:
    """Accuracy check comparing the routing maps semantically.

    The census scratch is kernel-internal, so only the layouts it feeds are
    observable: expert_ids must match byte-for-byte, while the within-expert
    slot order is not deterministic (the kernel claims ranks with atomics,
    the reference with a stable sort), so sorted_token_ids is compared as
    per-expert slot multisets -- an expert's slots are contiguous but the
    split across its blocks is arbitrary, so only the multiset is promised --
    and the permutation as: gathering each valid slot's row through inv_perm
    reproduces its token's source row.
    """
    if not isinstance(actual, tuple | list):
        assert actual is None
        return
    assert isinstance(expected, tuple | list)
    a, ref = list(actual), list(expected)
    # sorted(2) expert_ids(3) num_tokens(4) top_k(5) m_block_size(6) E(7)
    # local_E(8) src_aq(9) src_scale(10) permuted_aq(11) permuted_scale(12)
    # inv_perm(13) ntp(14) num_valid(15)
    for i in (14, 15):
        if a[i] is not None:
            assert torch.equal(a[i], ref[i])

    topk_ids = a[0]
    num_tokens, top_k = topk_ids.shape
    m_block_size = a[6]
    routed = num_tokens * top_k
    # Only the live blocks are written: the expert_ids -1 tail and the
    # worst-case buffer fill are gone (consumers prune blocks past the count),
    # so everything below compares that prefix.
    live_blocks = int(a[14]) // m_block_size if a[14] is not None else a[3].numel()
    assert torch.equal(a[3][:live_blocks], ref[3][:live_blocks])
    # Padding lanes inside live blocks are the only ones a GEMM may read beyond
    # the real slots, and they must carry the sentinel, not stale memory.
    live = a[2][: live_blocks * m_block_size]
    assert torch.equal(
        live[live >= routed], torch.full_like(live[live >= routed], routed)
    )
    sa, sr, ea = a[2], ref[2], a[3]

    def slots_by_expert(sorted_ids: torch.Tensor) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for blk in range(live_blocks):
            expert = int(ea[blk])
            if expert < 0:
                continue
            block = sorted_ids[blk * m_block_size : (blk + 1) * m_block_size]
            groups.setdefault(expert, []).extend(block[block < routed].tolist())
        return {expert: sorted(slots) for expert, slots in groups.items()}

    assert slots_by_expert(sa) == slots_by_expert(sr)

    if a[9] is not None:
        source_aq, source_scale = a[9], a[10]
        inv = a[13].to(torch.long)
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
            a[11][inv][valid_slots], source_aq[slots // top_k][valid_slots]
        )
        assert torch.equal(
            a[12][inv][valid_slots], source_scale[slots // top_k][valid_slots]
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
        "num_valid",
    ],
    config_picker=pick_config,
    input_generator=generate_inputs,
    fake_impl=fake_impl,
    helion_settings=helion.Settings(
        fast_math=True,
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
    num_experts: int,  # global expert count (EP=1: the local count)
    num_local_experts: int,  # experts this rank owns; pick_config's EP = E / it
    source_aq: torch.Tensor
    | None = None,  # [num_tokens, K] (in; None = skip permutation)
    source_scale: torch.Tensor | None = None,  # [num_tokens, K // 128] (in)
    permuted_aq: torch.Tensor | None = None,  # [routed, K] (out)
    permuted_scale: torch.Tensor | None = None,  # [routed, K // 128] (out)
    inv_perm: torch.Tensor | None = None,  # [routed] int32 (out)
    num_tokens_post_padded: torch.Tensor
    | None = None,  # [1] int32 (out: padded routed-slot count)
    num_valid: torch.Tensor | None = None,  # [1] int32 (out: on-rank slot count)
) -> None:
    routed = num_tokens * top_k

    hl.specialize(top_k)
    hl.specialize(num_experts)
    hl.specialize(num_local_experts)
    # The block size follows the GEMM's align_block_size, so it is a small set
    # of values; specializing it lets the padding tail be sized with an arange
    # instead of a one-iteration tile loop.
    hl.specialize(m_block_size)
    hl.specialize(routed)

    topk_flat = topk_ids.reshape(-1)
    assert topk_flat.numel() == routed
    assert topk_ids.dtype == torch.int32
    assert expert_ids.dtype == torch.int32
    # The layout block size is specialized above; a non-power-of-two value is
    # still fine (the arange pads to the next power of two and the tail fill
    # masks the extra lanes).
    assert m_block_size > 0

    # Census scratch, allocated fresh per launch: the census accumulates
    # atomically so it starts zeroed, pass 2 fully writes the offsets, and
    # ``ranks`` keeps each slot's claimed rank from pass 1 for pass 3 to use.
    # The caller sizes sorted_token_ids / expert_ids with the _align_sizes
    # formulas (worst case), but only the live range is ever written: the
    # consumers prune blocks at ``num_tokens_post_padded``, real slots are
    # scattered by pass 3, and pass 3 also writes the padding tails. Nothing
    # fills the buffer worst-case, so the region past the count is stale memory
    # and must stay unread.
    # One row past the experts is the census's scratch target for off-rank
    # slots: ``hl.atomic_add`` has no mask, so masked lanes are redirected to it
    # instead of being skipped. Keeping the census a helion op (rather than an
    # inline-Triton masked atomic) is what lets the dependency analysis see that
    # pass 1 writes this buffer and order the zero-fill below before it.
    counts_buf = torch.empty(num_experts + 1, dtype=torch.int32, device=topk_ids.device)
    ranks_buf = torch.empty(routed, dtype=torch.int32, device=topk_ids.device)
    offs_buf = torch.empty(num_experts, dtype=torch.int32, device=topk_ids.device)
    block_offs_buf = torch.empty(num_experts, dtype=torch.int32, device=topk_ids.device)

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

    # The caller sizes the layout buffers worst-case (mega's _align_sizes); the
    # kernel writes only the live blocks and their padding tails.
    max_padded = sorted_token_ids.shape[0]
    num_blocks = expert_ids.shape[0]
    assert max_padded >= routed
    assert num_blocks >= (max_padded + m_block_size - 1) // m_block_size

    hl.pdl_wait()
    hl.pdl_launch_dependents()

    for tile_e in hl.tile(num_experts):
        counts_buf[tile_e] = hl.zeros([tile_e], dtype=counts_buf.dtype)

    # Pass 1: census each slot's (locally mapped) expert, claiming its rank in
    # the same atomic -- its return value is the slot's rank within the expert
    # (the k-th claim on it), and the counter it leaves behind is the census
    # pass 2 reads. Nothing else is written here: the layout's padding is filled
    # where it is actually consumed (the placement pass's tail fill), and blocks
    # past the count are pruned by the consumers, so there is no worst-case fill.
    for tile_m in hl.tile(routed):
        e = hl.load(topk_flat, [tile_m], extra_mask=tile_m.index < routed)
        if expert_map is not None:
            e = expert_map[e].to(torch.int32)
        e = torch.where(tile_m.index < routed, e, -1)

        valid = e >= 0
        rank = hl.atomic_add(counts_buf, [torch.where(valid, e, num_experts)], 1)
        hl.store(ranks_buf, [tile_m], rank, extra_mask=valid)

    # Pass 2: exclusive per-expert offsets from the census, with four programs
    # (``hl.grid(4)``) each owning one output: the permutation row offsets and
    # the block offsets (prefix scans), then the padded routed-slot count and
    # the exact on-rank slot count (totals). Splitting them keeps a program's
    # work to a single E-wide pass -- the previous forms had one program issue
    # all four serially, and before that every program re-summed the whole
    # census, O(E^2) in all.
    for i in hl.grid(4):
        e_all = hl.arange(num_experts)
        counts = hl.load(counts_buf, [e_all], extra_mask=e_all < num_experts)
        mask = e_all < num_experts

        if i == 0:  # noqa: SIM102 - `and` would truth-test a traced condition
            if source_aq is not None:
                hl.store(
                    offs_buf,
                    [e_all],
                    hl.cumsum(counts, dim=0) - counts,
                    extra_mask=mask,
                )
        if i == 1:
            blocks_per_expert = (counts + (m_block_size - 1)) // m_block_size
            hl.store(
                block_offs_buf,
                [e_all],
                hl.cumsum(blocks_per_expert, dim=0) - blocks_per_expert,
                extra_mask=mask,
            )
        if i == 2:
            blocks_per_expert = (counts + (m_block_size - 1)) // m_block_size
            if num_tokens_post_padded is not None:
                total_blocks = blocks_per_expert[None, :].sum(dim=1)
                hl.store(
                    num_tokens_post_padded,
                    [hl.arange(1)],
                    total_blocks * m_block_size,
                )
        if i == 3:  # noqa: SIM102 - `and` would truth-test a traced condition
            if num_valid is not None:
                total_valid = counts[None, :].sum(dim=1)
                hl.store(num_valid, [hl.arange(1)], total_valid)

    # Pass 3: place every slot -- gather its expert's block offset and the
    # rank claimed in pass 1, then scatter. No atomics here: every ordering
    # decision was made in the census pass. The slot block is fixed so the
    # row-copy tiles stay small at untuned defaults.
    for tile_r in hl.tile(routed):
        # Cleared like the census above: a padded lane would otherwise read
        # expert 0's offsets and scatter its out-of-range slot id.
        e = topk_flat[tile_r]
        if expert_map is not None:
            e = expert_map[e].to(torch.int32)
        e = torch.where(tile_r.index < routed, e, -1)
        valid = e >= 0

        # The rank pass 1 claimed. Skipped lanes read an undefined value and
        # their -1 expert is never dereferenced (every access below is masked),
        # so no clamped copy of ``e`` is needed.
        rank = hl.load(ranks_buf, [tile_r])

        blk = hl.load(block_offs_buf, [e], extra_mask=valid) + rank // m_block_size
        hl.store(
            sorted_token_ids,
            [blk * m_block_size + rank % m_block_size],
            tile_r.index,
            extra_mask=valid,
        )
        hl.store(expert_ids, [blk], e, extra_mask=valid)

        # Sentinel-fill this expert's padding tail: its slots fill the blocks in
        # rank order, so the only padding a GEMM can read is
        # [counts[e], blocks_e * m_block_size), the tail of its last block. The
        # real slots live below counts[e], so the fill never races them, and
        # every slot of the expert writes the same lanes idempotently. This
        # replaces the worst-case fill over the whole padded buffer.
        cnt = hl.load(counts_buf, [e], extra_mask=valid)
        blocks_e = (cnt + m_block_size - 1) // m_block_size
        pad = blocks_e * m_block_size - cnt
        base = hl.load(block_offs_buf, [e], extra_mask=valid) * m_block_size
        pad_cols = hl.arange(m_block_size)
        tail = base[:, None] + cnt[:, None] + pad_cols[None, :]
        hl.store(
            sorted_token_ids,
            [tail],
            (tail * 0 + routed).to(sorted_token_ids.dtype),
            extra_mask=valid[:, None] & (pad_cols[None, :] < pad[:, None]),
        )
        if source_aq is not None:
            perm_row = hl.load(offs_buf, [e], extra_mask=valid) + rank

            hl.store(inv_perm, [tile_r], perm_row, extra_mask=valid)
            # Off-rank lanes read nothing: the load carries the same slot mask
            # as the store, so the source row needs no clamping (the tile's own
            # bounds mask covers lanes past the routed count).
            src = tile_r.index // top_k
            for tile_k in hl.tile(K):
                a = hl.load(source_aq, [src, tile_k], extra_mask=valid[:, None])
                hl.store(permuted_aq, [perm_row, tile_k], a, extra_mask=valid[:, None])
            for tile_g in hl.tile(num_group_k):
                sc = hl.load(source_scale, [src, tile_g], extra_mask=valid[:, None])
                hl.store(
                    permuted_scale, [perm_row, tile_g], sc, extra_mask=valid[:, None]
                )
