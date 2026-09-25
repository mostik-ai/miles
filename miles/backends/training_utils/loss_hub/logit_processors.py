from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch

from miles.backends.training_utils.cp_utils import allgather_cp_redistribute, get_logits_and_tokens_offset_with_cp
from miles.backends.training_utils.loss_hub.fused_log_probs import fused_log_probs_and_entropy
from miles.backends.training_utils.loss_hub.math_utils import calculate_log_probs_and_entropy
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.sampling_mask import build_local_sampling_mask
from miles.utils.sampling_mask import RolloutSamplingMask


def _flatten_logits(logits: torch.Tensor, args: Namespace, max_seq_lens: list[int] | None) -> torch.Tensor:
    """``[1, T, V]`` (thd) or ``[B, S, V]`` (bshd) model logits as ``[rows, V]``, a view."""
    assert len(logits.shape) == 3, f"{logits.shape}"
    if args.qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        return logits.squeeze(0)
    assert max_seq_lens is not None
    return logits.view(-1, logits.size(-1))


def _iter_response_rows(
    num_rows: int,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
    include_response_indices: bool,
) -> Iterator[tuple[tuple[tuple[int, int], ...], torch.Tensor, Sequence[int]]]:
    """Yield, per sample, the rows of the flattened logits that score its response tokens.

    When context parallelism is disabled, a response occupies one run of the concatenated sequence.
    With context parallelism, this rank holds part of it: one run under all-gather CP, two runs
    (the zigzag halves) otherwise.

    Args:
        num_rows: rows of the flattened ``[rows, V]`` logits on this rank.
        args: Configuration (``qkv_format``, ``allgather_cp``).
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(row_ranges, tokens_chunk, response_indices)`: half-open `(start, end)` row runs
        in order, whose rows score `tokens_chunk` (1D int64) one to one. `response_indices` maps
        every local row back to the full response; it is empty when `include_response_indices`
        is false.
    """
    qkv_format = args.qkv_format
    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
            else:
                end += total_length
            start = end - response_length
            row_ranges = ((start - 1, end - 1),)
            tokens_chunk = tokens[-response_length:] if response_length else tokens[0:0]
            response_indices = range(response_length) if include_response_indices else ()
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            chunk_start = parallel_state.cp.rank * num_rows
            chunk_end = chunk_start + num_rows

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                row_ranges = ((0, 0),)
                tokens_chunk = tokens[0:0]
                response_indices = ()
            else:
                row_ranges = ((s - chunk_start, e - chunk_start),)
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
                response_indices = (
                    range(
                        s - logit_global_start,
                        e - logit_global_start,
                    )
                    if include_response_indices
                    else ()
                )
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            row_ranges = tuple(
                (
                    base + logits_offset[half][0] - chunks_offset[half][0],
                    base + logits_offset[half][1] - chunks_offset[half][0],
                )
                for half, base in enumerate((end, end + chunk_size))
            )
            end += 2 * chunk_size

            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]
            for (row_start, row_end), tokens_half in zip(row_ranges, (tokens_0, tokens_1), strict=True):
                assert row_end - row_start == tokens_half.size(0), f"{row_end - row_start} vs {tokens_half.size(0)}"

            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)
            if include_response_indices:
                prompt_length = total_length - response_length
                response_indices = [
                    *range(
                        tokens_offset[0][0] - prompt_length,
                        tokens_offset[0][1] - prompt_length,
                    ),
                    *range(
                        tokens_offset[1][0] - prompt_length,
                        tokens_offset[1][1] - prompt_length,
                    ),
                ]
            else:
                response_indices = ()

        seq_start += total_length

        assert sum(row_end - row_start for row_start, row_end in row_ranges) == tokens_chunk.size(0)
        if include_response_indices:
            assert len(response_indices) == tokens_chunk.size(0)
        yield row_ranges, tokens_chunk, response_indices


def _iter_response_chunks(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
    include_response_indices: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, Sequence[int]]]:
    """Yield response logits, tokens, and original response indices per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample (see ``_iter_response_rows`` for the row layout).

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk, response_indices)`, where
        `logits_chunk` is shape `[R, V]` (policy) or `[R, 1]` (value), and
        `tokens_chunk` is shape `[R]` (1D int64). `response_indices` maps every
        local row back to the full response. The mapping is empty when
        `include_response_indices` is false.
    """
    if not args.true_on_policy_mode:
        # Model-precision callers hand native bf16/fp16 logits; chunks are upcast to fp32 downstream
        assert logits.dtype in (torch.float32, torch.bfloat16, torch.float16), f"{logits.dtype}"
    logits = _flatten_logits(logits, args, max_seq_lens)

    if args.true_on_policy_mode:
        if logits.size(-1) > 1 and args.rollout_temperature > 0 and args.rollout_temperature != 1.0:
            logits = logits.div(args.rollout_temperature)
        if getattr(args, "bf16", False):
            logits = logits.to(torch.bfloat16)
        elif getattr(args, "fp16", False):
            logits = logits.to(torch.float16)

    for row_ranges, tokens_chunk, response_indices in _iter_response_rows(
        logits.size(0),
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        include_response_indices=include_response_indices,
    ):
        pieces = [logits[row_start:row_end] for row_start, row_end in row_ranges]
        logits_chunk = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        yield logits_chunk, tokens_chunk, response_indices


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample."""
    for logits_chunk, tokens_chunk, _ in _iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        include_response_indices=False,
    ):
        yield logits_chunk, tokens_chunk


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    entropy_requires_grad: bool = True,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    rollout_sampling_mask: Sequence[RolloutSamplingMask] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy is computed and returned only
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `calculate_log_probs_and_entropy`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        entropy_requires_grad: If False, compute entropy as an observed metric
            without attaching it to the autograd graph.
        non_loss_data: Unused; kept for API compatibility.
        rollout_sampling_mask: One ``RolloutSamplingMask`` per sample,
            covering every response token.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    assert non_loss_data
    if rollout_sampling_mask is not None:
        for sample_index, (sampling_mask, response_length) in enumerate(
            zip(rollout_sampling_mask, response_lengths, strict=True)
        ):
            if len(sampling_mask) != response_length:
                raise ValueError(
                    f"sampling-mask length {len(sampling_mask)} != response length "
                    f"{response_length} for sample {sample_index}"
                )
    parallel_state = get_parallel_state()
    if uses_fused_log_probs(args) and rollout_sampling_mask is None:
        res = _fused_log_probs_and_entropy(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=with_entropy,
            entropy_requires_grad=entropy_requires_grad,
            max_seq_lens=max_seq_lens,
        )
        if args.allgather_cp:
            allgather_cp_redistribute(
                res,
                logits=logits,
                args=args,
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                max_seq_lens=max_seq_lens,
            )
        return res

    log_probs_list = []
    entropy_list = []
    response_chunks = _iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        include_response_indices=rollout_sampling_mask is not None,
    )
    for sample_index, (logits_chunk, tokens_chunk, response_indices) in enumerate(response_chunks):
        sampling_mask = None
        if rollout_sampling_mask is not None:
            sampling_mask = build_local_sampling_mask(
                logits_chunk,
                rollout_sampling_mask[sample_index],
                response_indices,
                tp_rank=parallel_state.tp.rank,
            )
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            parallel_state.tp.group,
            with_entropy=with_entropy,
            entropy_requires_grad=entropy_requires_grad,
            chunk_size=args.log_probs_chunk_size,
            true_on_policy=args.true_on_policy_mode,
            vocab_size=getattr(args, "vocab_size", None),
            sampling_mask=sampling_mask,
            temperature=1.0 if args.true_on_policy_mode else args.rollout_temperature,
            debug_unified_grad_fused_logprob=args.debug_unified_grad_fused_logprob,
        )

        log_probs_list.append(log_prob.squeeze(-1))
        if with_entropy:
            entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res


def uses_fused_log_probs(args: Namespace) -> bool:
    """Whether ``get_log_probs_and_entropy`` takes the fused path (``--log-probs-backend fused``).

    True-on-policy mode (bitwise parity with the rollout engine) and sampling-support replay keep
    the torch path. On the fused path the logits always get a gradient from the log-prob op, even
    on a rank that scores no row, so callers need no ``0 * logits.sum()`` graph anchor.
    """
    return (
        getattr(args, "log_probs_backend", "torch") == "fused"
        and not args.true_on_policy_mode
        and not getattr(args, "use_sampling_support_replay", False)
    )


def _fused_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool,
    entropy_requires_grad: bool,
    max_seq_lens: list[int] | None,
) -> dict[str, list[torch.Tensor]]:
    """All samples' response rows through one ``fused_log_probs_and_entropy`` call."""
    flat_logits = _flatten_logits(logits, args, max_seq_lens)
    row_ranges, targets, lengths = [], [], []
    for sample_ranges, tokens_chunk, _ in _iter_response_rows(
        flat_logits.size(0),
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        include_response_indices=False,
    ):
        row_ranges.extend(sample_ranges)
        targets.append(tokens_chunk)
        lengths.append(tokens_chunk.size(0))
    assert all(row_start >= 0 for row_start, _ in row_ranges), f"negative logits row in {row_ranges}"
    device = flat_logits.device
    rows = torch.cat([torch.arange(row_start, row_end, device=device) for row_start, row_end in row_ranges])
    log_probs, entropy = fused_log_probs_and_entropy(
        flat_logits,
        rows,
        torch.cat(targets).to(device),
        tp_group=get_parallel_state().tp.group,
        temperature=args.rollout_temperature,
        with_entropy=with_entropy,
        entropy_requires_grad=entropy_requires_grad,
        # the checkpointed loss replays its forward on these logits during backward
        inplace_backward=not args.recompute_loss_function,
    )
    res = {"log_probs": list(log_probs.split(lengths))}
    if with_entropy:
        res["entropy"] = list(entropy.split(lengths))
    return res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        # upcast (no-op for fp32) so value-head outputs stay fp32 even when logits arrive bf16
        value_list.append(logits_chunk.squeeze(-1).float())

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res
