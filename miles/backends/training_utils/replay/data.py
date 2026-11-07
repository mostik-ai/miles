from typing import Protocol

import torch

from miles.backends.training_utils.data.context_parallel import slice_with_cp
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, get_parallel_state
from miles.utils.replay_base import record_streams


class RegisterReplayListFunc(Protocol):
    def __call__(self, replay_list: list, replay_data: torch.Tensor, **kwargs) -> None: ...


def register_replay_list_sequential(replay_list, replay_data, **_kwargs):
    """Map replay streams to registered modules.

    Each replay records `replay_data[:, replay.stream_idx]` if `stream_idx` is
    set (used for sparse layer layouts under PP/VPP, where the global replay
    tensor contains more streams than this rank registered). Otherwise falls
    back to 1:1 enumeration order.
    """
    stream_indices = []
    for replay_idx, replay in enumerate(replay_list):
        stream_idx = replay.stream_idx if replay.stream_idx is not None else replay_idx
        if not 0 <= stream_idx < replay_data.shape[1]:
            raise AssertionError(
                f"replay stream_idx {stream_idx} out of range " f"(replay_data has {replay_data.shape[1]} streams)"
            )
        stream_indices.append(stream_idx)
    record_streams(replay_list, replay_data, stream_indices)


def fill_replay_data(
    *,
    args,
    models,
    data_iterator,
    num_microbatches,
    rollout_data,
    data_key: str,
    replay_list: list,
    register_replay_list_func: RegisterReplayListFunc,
    if_sp_region=True,
    indices_are_token_positions=False,
    sequence_first=False,
    kept_in_train_dump=False,
):
    """Load rollout replay tensors into module replay queues.

    `rollout_data[data_key]` contains one tensor per sample with shape
    `[num_tokens - 1, num_streams, topk]`. This function replays the training
    data iterator to process those tensors in the same microbatch order as
    log-prob and train forwards, pads/slices them to match the local CP/SP
    token layout, and then delegates stream-to-module mapping to
    `register_replay_list_func`.

    For `bshd`, `sequence_first` selects Megatron MoE's `[S, B]` token order;
    indexer replay consumes the default `[B, S]` order.
    """
    if data_key not in rollout_data:
        raise ValueError(f"{data_key} is required in rollout_data for replay.")

    for iterator in data_iterator:
        iterator.reset()

    parallel_state = get_parallel_state()
    qkv_format = args.qkv_format
    # sequence_parallel is Megatron-only; FSDP has tp_size == 1 so the slice is a no-op there.
    sequence_parallel = args.train_backend == "megatron" and args.backend.sequence_parallel and if_sp_region

    def pad_func(data, pad):
        _, num_layers, topk = data.shape
        pad_tensor = torch.full(
            (pad, num_layers, topk),
            fill_value=-1,
            device=data.device,
            dtype=data.dtype,
        )
        return torch.cat([data, pad_tensor], dim=0)

    for _ in range(sum(num_microbatches)):
        batch = data_iterator[0].get_next([data_key, "tokens", "max_seq_lens"])
        replay_data = batch[data_key]
        tokens = batch["tokens"]
        assert len(replay_data) == len(tokens)
        for a, b in zip(replay_data, tokens, strict=False):
            assert a.shape[0] == b.shape[0] - 1, f"{a.shape}, {b.shape}"

        if qkv_format == "bshd":
            replay_data = _local_bshd_rows(
                replay_data,
                parallel_state=parallel_state,
                max_seqlen=batch["max_seq_lens"][0],
                allgather_cp=args.allgather_cp,
                sequence_first=sequence_first,
                sequence_parallel=sequence_parallel,
            )
        else:
            # Pad replay data to align with the token batch's final token. The padded token is masked from loss.
            # TODO: fuse this padding with the following slice_with_cp to reduce memory copy.
            replay_data = [pad_func(r, 1) for r in replay_data]
            # TODO: maybe extract a common process function for here and get_batch?

            cp_size = parallel_state.cp.size
            cp_rank = parallel_state.cp.rank
            pad_size = parallel_state.tp.size * args.data_pad_size_multiplier
            if args.allgather_cp and cp_size > 1:
                replay_data = torch.cat(replay_data, dim=0)
                global_pad_size = cp_size * pad_size
                pad = (global_pad_size - replay_data.size(0) % global_pad_size) % global_pad_size
                if pad != 0:
                    replay_data = pad_func(replay_data, pad)
                replay_data = replay_data.chunk(cp_size, dim=0)[cp_rank]
            else:
                replay_data = [slice_with_cp(r, pad_func, qkv_format) for r in replay_data]
                if indices_are_token_positions:
                    # map indices to thd format
                    offset = 0
                    for i, r in enumerate(replay_data):
                        replay_data[i] = torch.where(r != -1, r + offset, r)
                        offset += r.shape[0]
                replay_data = torch.cat(replay_data, dim=0)
                pad = (pad_size - replay_data.size(0) % pad_size) % pad_size
                if pad != 0:
                    replay_data = pad_func(replay_data, pad)
            if sequence_parallel:
                replay_data = _sequence_parallel_rows(replay_data, parallel_state.tp)

        register_replay_list_func(replay_list, replay_data, models=models)

    # A train-data dump, written after training, keeps the column so it holds what replay consumed.
    if not (kept_in_train_dump and args.save_debug_train_data is not None):
        del rollout_data[data_key]

    for iterator in data_iterator:
        iterator.reset()


def _local_bshd_rows(
    replay_data: list[torch.Tensor],
    *,
    parallel_state: ParallelState,
    max_seqlen: int,
    allgather_cp: bool,
    sequence_first: bool,
    sequence_parallel: bool,
) -> torch.Tensor:
    """Copy only the rows of a bshd micro-batch that this rank's token layout holds.

    Each sample is padded with -1 rows to `max_seqlen`, the row of its final token included. The samples are laid out
    `[S, B]` when `sequence_first` and `[B, S]` otherwise, then sliced for context and sequence parallelism like the
    tokens. The layout is worked out on positions, so no rank copies whole padded samples.
    """
    longest = max(len(sample) for sample in replay_data)
    assert longest < max_seqlen, f"a sample has {longest + 1} tokens, more than max_seqlen {max_seqlen}"
    positions = _local_bshd_positions(max_seqlen, parallel_state=parallel_state, allgather_cp=allgather_cp)
    sample_indices = torch.arange(len(replay_data))
    if sequence_first:
        row_positions = positions.repeat_interleave(len(replay_data))
        row_samples = sample_indices.repeat(len(positions))
    else:
        row_positions = positions.repeat(len(replay_data))
        row_samples = sample_indices.repeat_interleave(len(positions))
    if sequence_parallel:
        row_positions = _sequence_parallel_rows(row_positions, parallel_state.tp)
        row_samples = _sequence_parallel_rows(row_samples, parallel_state.tp)

    _, num_streams, topk = replay_data[0].shape
    local_rows = replay_data[0].new_full((len(row_positions), num_streams, topk), -1)
    for sample_index, sample in enumerate(replay_data):
        holds_sample = (row_samples == sample_index) & (row_positions >= 0) & (row_positions < len(sample))
        rows = holds_sample.nonzero().squeeze(1)
        local_rows[rows] = sample[row_positions[rows]]
    return local_rows


def _local_bshd_positions(max_seqlen: int, *, parallel_state: ParallelState, allgather_cp: bool) -> torch.Tensor:
    """Positions of a padded bshd sample that this context-parallel rank holds; -1 marks zigzag padding."""
    cp = parallel_state.cp
    positions = torch.arange(max_seqlen)
    if cp.size == 1:
        return positions
    if allgather_cp:
        assert max_seqlen % cp.size == 0, f"max_seqlen {max_seqlen} must be divisible by cp_size {cp.size}"
        local_len = max_seqlen // cp.size
        return positions[cp.rank * local_len : (cp.rank + 1) * local_len]
    return slice_with_cp(positions, -1, "bshd", max_seqlen, parallel_state)


def _sequence_parallel_rows(rows: torch.Tensor, tp: GroupInfo) -> torch.Tensor:
    """The rows this tensor-parallel rank holds when sequence parallelism splits the first dimension."""
    num_rows = rows.size(0)
    assert num_rows % tp.size == 0
    width = num_rows // tp.size
    return rows[width * tp.rank : width * (tp.rank + 1)]
