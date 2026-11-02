import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles_plugins.models.cp_utils import packed_shard_to_zigzag, zigzag_to_packed_shard

register_cpu_ci(est_time=20, suite="stage-b-cpu", labels=["precision"])


def _stream_and_shards(world_size: int, rank: int, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A packed stream as Megatron packs it, sequences of 2 * world_size equal chunks and a tail padding split into
    world_size, with rows of distinct values; returns its boundaries, this rank's zigzag shard built chunk by chunk,
    and this rank's contiguous shard."""
    generator = torch.Generator().manual_seed(seed)
    lengths = [
        2 * world_size * int(chunk)
        for chunk in torch.randint(1, 8, (int(torch.randint(1, 6, (1,), generator=generator)),), generator=generator)
    ]
    lengths.append(world_size * int(torch.randint(1, 4, (1,), generator=generator)))
    total = sum(lengths)
    stream = torch.arange(total, dtype=torch.float64).unsqueeze(1) * torch.tensor(
        [1.0, -2.0, 0.5], dtype=torch.float64
    )
    cu_seqlens = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    zigzag = []
    for start, length in zip(cu_seqlens[:-1].tolist(), lengths, strict=True):
        if length % (2 * world_size):
            zigzag.append(stream[start : start + length].chunk(world_size)[rank])
        else:
            chunks = stream[start : start + length].chunk(2 * world_size)
            zigzag += [chunks[rank], chunks[2 * world_size - 1 - rank]]
    shard = total // world_size
    return cu_seqlens, torch.cat(zigzag), stream[rank * shard : (rank + 1) * shard]


def _check_relayouts(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        group = dist.group.WORLD
        for seed in range(5):
            cu_seqlens, zigzag, packed = _stream_and_shards(world_size, rank, seed)

            zigzag_leaf = zigzag.clone().requires_grad_(True)
            packed_shard = zigzag_to_packed_shard(zigzag_leaf, cu_seqlens, group, rank, world_size)
            (packed_shard * packed).sum().backward()
            torch.testing.assert_close(packed_shard, packed, rtol=0, atol=0)
            torch.testing.assert_close(zigzag_leaf.grad, zigzag, rtol=0, atol=0)

            packed_leaf = packed.clone().requires_grad_(True)
            zigzag_shard = packed_shard_to_zigzag(packed_leaf, cu_seqlens, group, rank, world_size)
            (zigzag_shard * zigzag).sum().backward()
            torch.testing.assert_close(zigzag_shard, zigzag, rtol=0, atol=0)
            torch.testing.assert_close(packed_leaf.grad, packed, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4])
def test_relayouts_between_zigzag_and_contiguous_shards_match_the_packed_stream_and_carry_gradients_back(world_size):
    run_multiprocess(_check_relayouts, world_size=world_size)
