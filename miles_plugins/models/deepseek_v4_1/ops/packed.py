"""Packed (thd) streams for DeepSeek-V4.1: each sequence attends to, compresses and indexes only its own tokens.

Compressed rows are sequence-major, as in DeepSeek-V4's thd path: the compressed rows of a sequence follow those of
the sequences before it. Indices into them are absolute, so a later layer that reuses a source layer's latents, index
keys or top-k needs no bookkeeping of its own.
"""

import itertools
from dataclasses import dataclass

import torch

from miles_plugins.models.deepseek_v4.ops.thd_utils import (
    ThdLayout,
    batch_of_row,
    compressed_cu_seqlens,
    get_q_positions_thd,
)


@dataclass(frozen=True)
class PackedSegment:
    """One sequence: its rows ``[row_start, row_end)`` of the stream and its compressed rows ``[key_start, key_end)``."""

    row_start: int
    row_end: int
    key_start: int
    key_end: int


@dataclass(frozen=True)
class PackedCompression:
    """The compressed rows of a packed stream at one compress ratio.

    ``token_rows`` holds the ``ratio`` stream rows each compressed row folds, or None at ratio 1, where every stream
    row is its own compressed row. ``positions`` is each compressed row's position in its own sequence, which its
    rotary embedding takes.
    """

    segments: tuple[PackedSegment, ...]
    token_rows: torch.Tensor | None
    positions: torch.Tensor


def packed_compression(thd: ThdLayout, *, ratio: int, total_rows: int) -> PackedCompression:
    """Each sequence's compressed rows at ``ratio``; a sequence's ``length % ratio`` trailing tokens fold into none."""
    keys = [length // ratio for length in thd.seq_lens]
    row_starts = list(itertools.accumulate(thd.seq_lens, initial=0))
    key_starts = list(itertools.accumulate(keys, initial=0))
    segments = tuple(
        PackedSegment(row_starts[i], row_starts[i + 1], key_starts[i], key_starts[i + 1]) for i in range(len(keys))
    )
    if ratio == 1:
        return PackedCompression(segments, token_rows=None, positions=get_q_positions_thd(thd.cu_seqlens, total_rows))
    n_rows = key_starts[-1]
    cu_compressed = compressed_cu_seqlens(thd.cu_seqlens, ratio)
    sequence = batch_of_row(cu_compressed, n_rows)
    local = torch.arange(n_rows, device=thd.cu_seqlens.device) - cu_compressed[sequence]
    first = thd.cu_seqlens[sequence] + local * ratio
    token_rows = first.unsqueeze(1) + torch.arange(ratio, device=first.device)
    return PackedCompression(segments, token_rows=token_rows, positions=local * ratio)
