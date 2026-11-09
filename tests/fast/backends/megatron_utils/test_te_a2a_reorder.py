import random

import pytest
import torch

from miles.backends.megatron_utils.te_a2a_reorder import reorder_after_a2a_before_attn, reorder_before_a2a_after_attn

context_parallel = pytest.importorskip("transformer_engine.pytorch.attention.dot_product_attention.context_parallel")
SEQ_CHUNK_IDS = {
    2: [0, 2, 3, 1],
    4: [0, 2, 4, 6, 7, 5, 3, 1],
    8: [0, 2, 4, 6, 8, 10, 12, 14, 15, 13, 11, 9, 7, 5, 3, 1],
}


def packed_sequences(seed: int, cp_size: int) -> torch.Tensor:
    rng = random.Random(seed)
    lengths = [2 * cp_size * rng.randint(1, 40) for _ in range(rng.randint(1, 9))]
    return torch.tensor([0, *lengths]).cumsum(0)


@pytest.mark.parametrize("cp_size", [2, 4, 8])
@pytest.mark.parametrize("seed", range(5))
def test_the_indices_match_transformer_engines_own(cp_size: int, seed: int) -> None:
    cu_seqlens = packed_sequences(seed, cp_size)
    x = torch.arange(int(cu_seqlens[-1])).float().unsqueeze(1)
    chunk_ids = torch.tensor(SEQ_CHUNK_IDS[cp_size])

    assert torch.equal(
        reorder_before_a2a_after_attn(x, cu_seqlens, cp_size),
        context_parallel.reorder_seq_chunks_before_a2a_after_attn_thd(x, cu_seqlens, cp_size),
    )
    assert torch.equal(
        reorder_after_a2a_before_attn(x, cu_seqlens, chunk_ids, cp_size),
        context_parallel.reorder_seq_chunks_after_a2a_before_attn_thd(x, cu_seqlens, chunk_ids, cp_size),
    )


def test_the_documented_example_reorders_as_documented() -> None:
    x = torch.tensor([*range(8), *range(8), *range(8), *range(16)]).float()
    cu_seqlens = torch.tensor([0, 8, 16, 24, 40])

    reordered = reorder_before_a2a_after_attn(x, cu_seqlens, 4)

    assert reordered.tolist()[:10] == [0, 7, 0, 7, 0, 7, 0, 1, 14, 15]
