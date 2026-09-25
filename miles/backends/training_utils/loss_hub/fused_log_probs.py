"""Fused per-token log-probability and entropy over vocab-parallel logits.

Every Miles loss reads the policy through the log-probability of each sampled token, plus an
optional entropy term (``get_log_probs_and_entropy``). The unfused path copies each response chunk
to fp32 and keeps the fused cross-entropy's fp32 softmax alive until backward; an entropy gradient
keeps another fp32 copy and softmax. At 128K tokens per rank that is about 95 GiB for the
log-probs alone.

This op streams each selected row once and keeps three fp32 numbers per row: the log-sum-exp, the
target logit and the softmax mean of the logits. Tensor-parallel ranks combine them with one max
and one sum all-reduce over ``[rows, 3]``. The backward streams the rows again, recomputes the
softmax from the saved log-sum-exp, and writes

    dlogits = (g * (onehot(y) - p) - c * p * (z - mu)) / T

where ``z = logits / T``, ``g`` and ``c`` are the upstream gradients of the log-prob and the
entropy, and ``mu`` is the softmax mean of ``z``. With ``inplace_backward`` it writes into the
logits buffer itself and zeroes the rows it did not score, so the backward allocates nothing of
vocab size.

CUDA tensors use the Triton kernels in ``fused_log_probs_triton``; CPU tensors use the same math in
torch, which keeps the autograd and tensor-parallel logic testable without a GPU.
"""

import torch
import torch.distributed as dist
from torch import Tensor

_NO_ENTROPY, _ENTROPY_METRIC, _ENTROPY_WITH_GRAD = 0, 1, 2


def fused_log_probs_and_entropy(
    logits: Tensor,
    rows: Tensor,
    targets: Tensor,
    *,
    tp_group: dist.ProcessGroup | None,
    temperature: float = 1.0,
    with_entropy: bool = False,
    entropy_requires_grad: bool = True,
    inplace_backward: bool = False,
) -> tuple[Tensor, Tensor | None]:
    """Log-prob of ``targets`` and entropy at the selected ``rows`` of ``logits / temperature``.

    Args:
        logits: ``[N, V_local]`` vocab-parallel logits in any float dtype, last dim contiguous.
        rows: ``[R]`` int64 row indices into ``logits``, each at most once.
        targets: ``[R]`` int64 token ids in the full vocabulary.
        tp_group: the vocab-parallel group; ``None`` for an unsplit vocabulary.
        temperature: divides the logits; values ``<= 0`` mean no scaling, as in the torch path.
        with_entropy: also return the entropy of each row.
        entropy_requires_grad: when False the entropy is a metric and carries no gradient.
        inplace_backward: write the logits gradient into ``logits`` itself. Only safe when nothing
            else reads the logits after this op's backward.

    Returns:
        ``(log_probs, entropy)`` fp32 ``[R]`` tensors; ``entropy`` is None unless requested.
    """
    assert logits.dim() == 2 and logits.stride(-1) == 1, f"need [N, V] logits, got {tuple(logits.shape)}"
    assert rows.shape == targets.shape, f"{tuple(rows.shape)} rows vs {tuple(targets.shape)} targets"
    temperature = float(temperature) if temperature > 0 else 1.0
    rows, targets = rows.long(), targets.long()

    if not (torch.is_grad_enabled() and logits.requires_grad):
        lse, target_logit, mu = _row_statistics(logits, rows, targets, tp_group, temperature, with_entropy)
        return target_logit - lse, (lse - mu if with_entropy else None)

    mode = _NO_ENTROPY if not with_entropy else (_ENTROPY_WITH_GRAD if entropy_requires_grad else _ENTROPY_METRIC)
    log_probs, entropy = _FusedLogProbsAndEntropy.apply(
        logits, rows, targets, tp_group, temperature, mode, inplace_backward
    )
    return log_probs, (entropy if with_entropy else None)


class _FusedLogProbsAndEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, rows, targets, tp_group, temperature, mode, inplace_backward):
        lse, target_logit, mu = _row_statistics(logits, rows, targets, tp_group, temperature, mode != _NO_ENTROPY)
        log_probs = target_logit - lse
        entropy = lse - mu if mode != _NO_ENTROPY else lse.new_empty(0)
        if mode != _ENTROPY_WITH_GRAD:
            ctx.mark_non_differentiable(entropy)
        ctx.save_for_backward(logits, rows, targets, lse, mu if mode == _ENTROPY_WITH_GRAD else None)
        ctx.tp_group = tp_group
        ctx.temperature = temperature
        ctx.mode = mode
        ctx.inplace_backward = inplace_backward
        return log_probs, entropy

    @staticmethod
    def backward(ctx, grad_log_probs, grad_entropy):
        logits, rows, targets, lse, mu = ctx.saved_tensors
        if ctx.mode != _ENTROPY_WITH_GRAD:
            grad_entropy = None
        grad = logits if ctx.inplace_backward else torch.empty_like(logits)
        _write_logits_grad(
            grad,
            logits,
            rows,
            targets,
            lse,
            mu,
            _contiguous_or_none(grad_log_probs),
            _contiguous_or_none(grad_entropy),
            vocab_start=_vocab_start(logits, ctx.tp_group),
            temperature=ctx.temperature,
        )
        scored = torch.zeros(logits.size(0), dtype=torch.bool, device=logits.device)
        scored[rows] = True
        grad.masked_fill_(~scored.unsqueeze(1), 0)
        return grad, None, None, None, None, None, None


def _row_statistics(logits, rows, targets, tp_group, temperature, with_entropy):
    """``(lse, target_logit, mu or None)`` per row, combined over the vocab-parallel group."""
    vocab_start = _vocab_start(logits, tp_group)
    if logits.is_cuda:
        from miles.backends.training_utils.loss_hub import fused_log_probs_triton  # needs triton

        row_max, row_sum, row_zsum, target_logit = fused_log_probs_triton.row_statistics(
            logits, rows, targets, vocab_start=vocab_start, temperature=temperature, with_entropy=with_entropy
        )
    else:
        row_max, row_sum, row_zsum, target_logit = _row_statistics_torch(
            logits, rows, targets, vocab_start=vocab_start, temperature=temperature, with_entropy=with_entropy
        )
    row_max, row_sum, row_zsum, target_logit = _combine_over_tp(row_max, row_sum, row_zsum, target_logit, tp_group)
    lse = row_max + torch.log(row_sum)
    return lse, target_logit, (row_zsum / row_sum if with_entropy else None)


def _combine_over_tp(row_max, row_sum, row_zsum, target_logit, tp_group):
    """Rescale each shard's sums to the global max and add them; one shard holds each target."""
    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return row_max, row_sum, row_zsum, target_logit
    global_max = row_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
    rescale = torch.exp(row_max - global_max)
    columns = [row_sum * rescale, target_logit] + ([row_zsum * rescale] if row_zsum is not None else [])
    sums = torch.stack(columns, dim=1)
    dist.all_reduce(sums, group=tp_group)
    return global_max, sums[:, 0], (sums[:, 2] if row_zsum is not None else None), sums[:, 1]


def _write_logits_grad(
    grad, logits, rows, targets, lse, mu, grad_log_probs, grad_entropy, *, vocab_start, temperature
):
    if logits.is_cuda:
        from miles.backends.training_utils.loss_hub import fused_log_probs_triton  # needs triton

        fused_log_probs_triton.write_logits_grad(
            grad,
            logits,
            rows,
            targets,
            lse,
            mu,
            grad_log_probs,
            grad_entropy,
            vocab_start=vocab_start,
            temperature=temperature,
        )
        return
    _write_logits_grad_torch(
        grad,
        logits,
        rows,
        targets,
        lse,
        mu,
        grad_log_probs,
        grad_entropy,
        vocab_start=vocab_start,
        temperature=temperature,
    )


def _row_statistics_torch(logits, rows, targets, *, vocab_start, temperature, with_entropy):
    z = logits.index_select(0, rows).float() / temperature
    row_max = z.max(dim=-1).values if rows.numel() else z.new_empty(0)
    e = torch.exp(z - row_max.unsqueeze(1))
    local_targets = targets - vocab_start
    in_shard = (local_targets >= 0) & (local_targets < logits.size(1))
    target_z = z.gather(1, local_targets.clamp(0, logits.size(1) - 1).unsqueeze(1)).squeeze(1)
    target_logit = torch.where(in_shard, target_z, torch.zeros_like(target_z))
    return row_max, e.sum(dim=-1), ((e * z).sum(dim=-1) if with_entropy else None), target_logit


def _write_logits_grad_torch(
    grad, logits, rows, targets, lse, mu, grad_log_probs, grad_entropy, *, vocab_start, temperature
):
    z = logits.index_select(0, rows).float() / temperature
    p = torch.exp(z - lse.unsqueeze(1))
    dz = torch.zeros_like(z)
    if grad_log_probs is not None:
        onehot = torch.zeros_like(z)
        local_targets = targets - vocab_start
        in_shard = (local_targets >= 0) & (local_targets < logits.size(1))
        onehot[in_shard.nonzero().squeeze(1), local_targets[in_shard]] = 1.0
        dz = grad_log_probs.unsqueeze(1) * (onehot - p)
    if grad_entropy is not None:
        dz = dz - grad_entropy.unsqueeze(1) * p * (z - mu.unsqueeze(1))
    grad[rows] = (dz / temperature).to(grad.dtype)


def _vocab_start(logits: Tensor, tp_group: dist.ProcessGroup | None) -> int:
    """First vocab id of this rank's shard: Megatron splits the padded vocab into equal shards."""
    if tp_group is None:
        return 0
    return dist.get_rank(tp_group) * logits.size(1)


def _contiguous_or_none(tensor: Tensor | None) -> Tensor | None:
    return tensor.contiguous() if tensor is not None else None
