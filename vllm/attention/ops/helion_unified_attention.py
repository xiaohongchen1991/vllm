# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import helion
import helion.language as hl
import torch

from .triton_unified_attention import (
    unified_attention as triton_baseline_unified_attention,
)


def _triton_baseline_fn(
    t_output,  # [num_tokens, num_query_heads, head_size]
    t_query,  # [num_tokens, num_query_heads, head_size]
    t_key_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_value_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_block_tables,  # [num_seqs, max_num_blocks_per_seq]
    t_seq_lens,  # [num_seqs]
    scale,
    # k_scale,
    # v_scale,
    t_query_start_lens,  # [num_seqs+1]
    # max_query_len,
    num_seqs,
):
    max_seqlen = t_seq_lens.max()
    max_query_len = t_query_start_lens.diff().max()
    return triton_baseline_unified_attention(
        q=t_query,
        k=t_key_cache,
        v=t_value_cache,
        out=t_output,
        cu_seqlens_q=t_query_start_lens,
        max_seqlen_q=max_query_len,
        seqused_k=t_seq_lens,
        max_seqlen_k=max_seqlen,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=t_block_tables,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )


nv_config_3d = helion.Config(
    block_sizes=[16, 1, 32],
    indexing=[
        "pointer",
        "pointer",
        "pointer",
        "pointer",
        "pointer",
        "pointer",
        "tensor_descriptor",
        "pointer",
    ],
    l2_groupings=[32],
    load_eviction_policies=["last", "last", "first", "last", "last", "last", "first"],
    loop_orders=[[0, 1], [1, 0]],
    num_stages=8,
    num_warps=2,
    pid_type="persistent_interleaved",
    range_flattens=[True, False, False, True],
    range_multi_buffers=[True, False, False, False],
    range_unroll_factors=[2, 2, 1, 0],
    range_warp_specializes=[False, None, False, True],
)
nv_config_2d = helion.Config(
    block_sizes=[16, 16],
    indexing=[
        "pointer",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "pointer",
    ],
    l2_groupings=[4],
    load_eviction_policies=["first", "", "last", "last", "first", "last", "last"],
    loop_orders=[[1, 0], [1, 0]],
    num_stages=2,
    num_warps=1,
    pid_type="persistent_interleaved",
    range_flattens=[True, None, True, False],
    range_multi_buffers=[True, None, None, None],
    range_unroll_factors=[3, 2, 2, 0],
    range_warp_specializes=[None, False, False, True],
)
amd_config = helion.Config(
    block_sizes=[32, 8],
    indexing=[
        "tensor_descriptor",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "pointer",
        "pointer",
        "pointer",
        "pointer",
    ],
    l2_groupings=[1],
    load_eviction_policies=["", "", "", "", "", "", ""],
    loop_orders=[[2, 1, 0], [0, 1]],
    num_stages=1,
    num_warps=4,
    pid_type="flat",
    range_flattens=[None, None, None, None],
    range_multi_buffers=[None, None, None, None],
    range_num_stages=[],
    range_unroll_factors=[0, 0, 0, 0],
    range_warp_specializes=[],
)

config = nv_config_3d


@helion.kernel(
    allow_warp_specialize=True,
    # dot_precision='ieee',
    config=config,
    autotune_baseline_fn=_triton_baseline_fn,
    autotune_effort="quick",
    static_shapes=False,
    print_output_code=False,
    print_repro=False,
    index_dtype=torch.int64,
)
def kernel_helion_attention(
    t_output,  # [num_tokens, num_query_heads, head_size]
    t_query,  # [num_tokens, num_query_heads, head_size]
    t_key_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_value_cache,  # [num_blks, blk_size, num_kv_heads, head_size]
    t_block_tables,  # [num_seqs, max_num_blocks_per_seq]
    t_seq_lens,  # [num_seqs]
    scale,
    # k_scale,
    # v_scale,
    t_query_start_lens,  # [num_seqs+1]
    # max_query_len,  # must be on cpu
    num_seqs,  # on cpu?
    # max_used_querylen_padded: hl.constexpr,
):
    head_size = hl.specialize(t_query.size(2))
    num_kv_heads = hl.specialize(t_key_cache.size(2))
    num_query_heads = hl.specialize(t_query.size(1))
    page_size = hl.specialize(t_value_cache.size(1))
    num_queries_per_kv = hl.specialize(num_query_heads // num_kv_heads)

    assert page_size == t_key_cache.size(1)
    assert head_size == t_key_cache.size(3)

    t_key_cache = t_key_cache.flatten(start_dim=0, end_dim=1)
    t_value_cache = t_value_cache.flatten(start_dim=0, end_dim=1)

    for seq_tile, kv_head_tile in hl.tile([num_seqs, num_kv_heads], block_size=[1, 1]):
        seq_idx = seq_tile.begin
        kv_head_idx = kv_head_tile.begin
        seq_len = t_seq_lens[seq_idx]
        query_start = t_query_start_lens[seq_idx]
        query_end = t_query_start_lens[seq_idx + 1]
        query_len = query_end - query_start
        context_len = seq_len - query_len

        head_start = kv_head_idx * num_queries_per_kv
        head_end = (kv_head_idx + 1) * num_queries_per_kv

        for tile_q, tile_h in hl.tile(
            [query_start, head_start],
            [query_end, head_end],
        ):
            # tile_m: tile_q x tile_h
            block_m_size = tile_h.block_size * tile_q.block_size
            query_pos = tile_q.index - query_start
            query_pos = (
                query_pos[:, None]
                .expand(tile_q.block_size, tile_h.block_size)
                .reshape(block_m_size)
            )

            # (tile_q, tile_h, HEAD_SIZE)
            # # tile_q is masked here
            q = t_query[tile_q, tile_h, :]
            # (tile_m, HEAD_SIZE)
            q = q.flatten(start_dim=0, end_dim=1)

            M = hl.full([block_m_size], float("-inf"), dtype=torch.float32)
            L = hl.full([block_m_size], 1.0, dtype=torch.float32)
            acc = hl.zeros([block_m_size, head_size], dtype=torch.float32)

            # adjust for causal mask
            max_seq_prefix_len = context_len + tile_q.end - query_start
            max_seq_prefix_len = torch.minimum(max_seq_prefix_len, seq_len)

            for tile_n in hl.tile(max_seq_prefix_len):
                block_n_size = tile_n.block_size
                key_idx = tile_n.index
                page_idx = key_idx // page_size  # [block_n_size]
                page_offset = key_idx % page_size  # [block_n_size]
                blk_idx = (
                    t_block_tables[seq_idx, page_idx]
                    .view([block_n_size])
                    .to(torch.int64)
                )  # [block_n_size]
                cache_idx = blk_idx * page_size + page_offset
                # (tile_n, HEAD_SIZE)
                k = t_key_cache[cache_idx, kv_head_idx, :]
                v = t_value_cache[cache_idx, kv_head_idx, :]
                # (HEAD_SIZE, tile_n)
                k = k.transpose(0, 1)
                # (tile_m, tile_n)
                S = hl.zeros([block_m_size, block_n_size], dtype=torch.float32)
                S = scale * hl.dot(q, k, out_dtype=torch.float32, acc=S)

                # causal mask
                causal_mask = key_idx[None, :] < context_len + query_pos[:, None] + 1
                S = torch.where(causal_mask, S, float("-inf"))

                M_j = torch.maximum(M, torch.amax(S, 1))
                # (tile_m, tile_n)
                P = torch.exp(S - M_j[:, None])
                # (tile_m, )
                L_j = torch.sum(P, 1)
                # (tile_m, )
                alpha = torch.exp(M - M_j)
                # (tile_m, HEAD_SIZE)
                acc = acc * alpha[:, None]
                L = L * alpha + L_j
                M = M_j

                # (tile_m, HEAD_SIZE)
                acc = hl.dot(P.to(v.dtype), v, out_dtype=torch.float32, acc=acc)

            # epilogue
            acc = acc / L[:, None]
            t_output[tile_q, tile_h, :] = acc.view(
                [tile_q.block_size, tile_h.block_size, head_size]
            )


def helion_unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    # max_query_len_int: int,
    num_seqs: int,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    assert alibi_slopes is None, "not supported right now, still experimental"
    assert softcap == 0, "not supported right now, still experimental"
    assert k_descale is None, "not supported right now, still experimental"
    assert v_descale is None, "not supported right now, still experimental"
    assert window_size == (-1, -1), "not supported right now, still experimental"

    block_size = v.shape[1]
    assert q.element_size() >= 2 or block_size >= 32, (
        "Block size must be at least 32 for fp8"
    )

    # max_used_querylen_padded = max_query_len_int if max_query_len_int == 1
    #   else torch._inductor.runtime.runtime_utils.next_power_of_2(
    #     max(16, max_query_len_int))

    kernel_helion_attention(
        t_output=out,
        t_query=q,
        t_key_cache=k,
        t_value_cache=v,
        t_block_tables=block_table,
        t_seq_lens=seqused_k,
        scale=softmax_scale,
        # k_scale=k_descale,
        # v_scale=v_descale,
        t_query_start_lens=cu_seqlens_q,
        # max_query_len=max_query_len_int,  # need not to be a tensor
        # max_used_querylen_padded = int(max_used_querylen_padded),
        num_seqs=num_seqs,
    )
