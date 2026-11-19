"""CPU tests for DeepSeek-V4.1's packed (thd) compressed rows.

A packed sequence must compress exactly the tokens it would compress alone, at the positions it would give them, and
its compressed rows must follow those of the sequences before it.
"""

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles_plugins.models.deepseek_v4.ops.thd_utils import ThdLayout
from miles_plugins.models.deepseek_v4_1.ops.packed import PackedSegment, packed_compression

register_cpu_ci(est_time=1, suite="stage-a-cpu", labels=[])

SHAPES = {
    "odd_tails": [1537, 77, 2050],
    "shorter_than_ratio": [5, 1, 9],
    "single": [300],
    "trailing_padding": [128, 63, 1],
}


def _layout(lens):
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    return ThdLayout(cu_seqlens=cu, seq_lens=tuple(lens), global_start=0, max_seqlen=max(lens))


def _alone(lens, ratio):
    """Each sequence's compressed rows as it compresses alone, shifted to where it sits in the stream."""
    token_rows, positions, segments = [], [], []
    row_start = key_start = 0
    for length in lens:
        groups = length // ratio
        token_rows += [[row_start + g * ratio + i for i in range(ratio)] for g in range(groups)]
        positions += [g * ratio for g in range(groups)]
        segments.append(PackedSegment(row_start, row_start + length, key_start, key_start + groups))
        row_start, key_start = row_start + length, key_start + groups
    return token_rows, positions, tuple(segments)


@pytest.mark.parametrize("lens", SHAPES.values(), ids=SHAPES.keys())
def test_ratio_two_groups_tokens_within_each_sequence(lens):
    compression = packed_compression(_layout(lens), ratio=2, total_rows=sum(lens))
    token_rows, positions, segments = _alone(lens, 2)
    assert compression.segments == segments
    assert compression.token_rows.tolist() == token_rows
    assert compression.positions.tolist() == positions


@pytest.mark.parametrize("lens", SHAPES.values(), ids=SHAPES.keys())
def test_ratio_one_keeps_every_row_at_its_position_in_its_sequence(lens):
    compression = packed_compression(_layout(lens), ratio=1, total_rows=sum(lens))
    _, positions, segments = _alone(lens, 1)
    assert compression.segments == segments
    assert compression.token_rows is None
    assert compression.positions.tolist() == positions


def test_rows_past_the_last_sequence_get_no_compressed_row():
    lens = [6, 3]
    compression = packed_compression(_layout(lens), ratio=2, total_rows=12)
    assert compression.token_rows.max().item() < sum(lens)
    assert compression.segments[-1].key_end == 4
