"""Around each all-to-all of its Ulysses path, Transformer Engine reorders sequence chunks with indices it rebuilds
sequence by sequence from 0-dim GPU tensors, so one attention call synchronizes with the GPU hundreds of times, about 9%
of a 9B training step:
https://github.com/NVIDIA/TransformerEngine/blob/release_v2.17/transformer_engine/pytorch/attention/dot_product_attention/context_parallel.py
These functions compute the same indices from plain integers and keep them for the micro-batch, whose attention layers
all ask for the same ones.
"""

import functools
import itertools

import torch

_CACHED_INDICES = 16


# TODO: delete this module when the image's Transformer Engine wheel no longer builds these indices from GPU scalars.
def patch_te_a2a_reorder() -> None:
    from transformer_engine.pytorch.attention.dot_product_attention import context_parallel

    context_parallel.reorder_seq_chunks_before_a2a_after_attn_thd = reorder_before_a2a_after_attn
    context_parallel.reorder_seq_chunks_after_a2a_before_attn_thd = reorder_after_a2a_before_attn


def reorder_before_a2a_after_attn(
    x: torch.Tensor, cu_seqlens: torch.Tensor, cp_size: int, seq_dim: int = 0
) -> torch.Tensor:
    return x.index_select(seq_dim, _before_a2a_indices(tuple(cu_seqlens.tolist()), cp_size, cu_seqlens.device))


def reorder_after_a2a_before_attn(
    x: torch.Tensor, cu_seqlens: torch.Tensor, seq_chunk_ids: torch.Tensor, cp_size: int, seq_dim: int = 0
) -> torch.Tensor:
    indices = _after_a2a_indices(tuple(cu_seqlens.tolist()), tuple(seq_chunk_ids.tolist()), cp_size, cu_seqlens.device)
    return x.index_select(seq_dim, indices)


@functools.lru_cache(maxsize=_CACHED_INDICES)
def _before_a2a_indices(cu_seqlens: tuple[int, ...], cp_size: int, device: torch.device) -> torch.Tensor:
    slices = 2 * cp_size
    segments = []
    for rank in range(cp_size):
        for start, end in itertools.pairwise(cu_seqlens):
            size = (end - start) // slices
            segments.append(torch.arange(start + rank * size, start + (rank + 1) * size))
            segments.append(torch.arange(start + (slices - rank - 1) * size, start + (slices - rank) * size))
    return torch.cat(segments).to(device)


@functools.lru_cache(maxsize=_CACHED_INDICES)
def _after_a2a_indices(
    cu_seqlens: tuple[int, ...], seq_chunk_ids: tuple[int, ...], cp_size: int, device: torch.device
) -> torch.Tensor:
    per_rank = [length // cp_size for length in cu_seqlens]
    segments = []
    for start, end in itertools.pairwise(per_rank):
        middle = (start + end) // 2
        for location, chunk_id in enumerate(seq_chunk_ids):
            offset = per_rank[-1] * (chunk_id // 2)
            first, last = (start, middle) if location < cp_size else (middle, end)
            segments.append(torch.arange(first + offset, last + offset))
    return torch.cat(segments).to(device)
