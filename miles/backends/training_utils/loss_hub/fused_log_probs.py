"""Fused per-token log-probability and entropy over vocab-parallel logits.

Every Miles loss reads the policy through the log-probability of each sampled token, plus an
optional entropy term (``get_log_probs_and_entropy``). The unfused path copies each response chunk
to fp32 and keeps the fused cross-entropy's fp32 softmax alive until backward; an entropy gradient
keeps another fp32 copy and softmax. At the 128K-decode shape (``[65536, 129280]`` bf16 logits on
one rank) that is 63 GiB for the log-probs and 79 GiB with an entropy gradient.

This op streams each selected row once and keeps a few fp32 numbers per row, all measured from the
row's max logit ``m`` in units of ``1 / T``: with ``d = (x - m) / T``, the log of ``S = sum exp(d)``,
the target's ``d_y``, and, for entropy, the softmax mean ``mu`` of ``d``. The log-prob is then
``d_y - log S`` and the entropy ``log S - mu``: for a confident token both are as accurate as
``torch.log_softmax``, where subtracting a log-sum-exp from the target logit would lose them to
cancellation. Tensor-parallel ranks combine the statistics with one max and one sum all-reduce over
``[rows, 2 or 3]``. The backward streams the rows again, recomputes ``p = exp(d - log S)``, and
writes

    dlogits = (g * (onehot(y) - p) - c * p * (d - mu)) / T

where ``g`` and ``c`` are the upstream gradients of the log-prob and the entropy; at the target,
``1 - p`` comes from ``expm1`` of the log-prob. With ``inplace_backward`` it writes into the logits
buffer itself and zeroes the rows it did not score, so the backward allocates nothing of vocab size.

The no-grad scoring passes (reference, old actor, teacher) call the same autograd function, like
every other Miles op: under ``torch.no_grad`` autograd keeps nothing, and the pass runs the
training forward's statistics kernel, so the stored old log-probs match the training forward bit
for bit. With more than two tensor-parallel ranks that holds up to the all-reduce: NCCL may add the
shards in another order when the two passes send different row counts.

CUDA tensors use the Triton kernels in ``fused_log_probs_triton``; CPU tensors use the same math in
torch, which keeps the autograd and tensor-parallel logic testable without a GPU.
"""

from typing import NamedTuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.autograd.function import once_differentiable


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

    log_probs, entropy = _FusedLogProbsAndEntropy.apply(
        logits,
        rows,
        targets,
        tp_group,
        temperature,
        with_entropy,
        with_entropy and entropy_requires_grad,
        inplace_backward,
    )
    return log_probs, (entropy if with_entropy else None)


class _RowStats(NamedTuple):
    """Per selected row, over the whole vocabulary, with ``d = (x - m) / T`` for the row max ``m``."""

    max: Tensor  # m
    log_sum: Tensor  # log sum_v exp(d_v)
    target: Tensor  # d_y
    mean: Tensor | None  # sum_v p_v d_v, only with entropy


class _FusedLogProbsAndEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, logits, rows, targets, tp_group, temperature, with_entropy, entropy_requires_grad, inplace_backward
    ):
        stats = _row_statistics(logits, rows, targets, tp_group, temperature, with_entropy)
        log_probs = stats.target - stats.log_sum
        entropy = stats.log_sum - stats.mean if with_entropy else log_probs.new_empty(0)
        if not entropy_requires_grad:
            ctx.mark_non_differentiable(entropy)
        # an output the loss does not use arrives as None, so the backward skips its term
        ctx.set_materialize_grads(False)
        ctx.save_for_backward(
            logits, rows, targets, stats.max, stats.log_sum, log_probs, stats.mean if entropy_requires_grad else None
        )
        ctx.tp_group = tp_group
        ctx.temperature = temperature
        ctx.entropy_requires_grad = entropy_requires_grad
        ctx.inplace_backward = inplace_backward
        return log_probs, entropy

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_log_probs, grad_entropy):
        logits, rows, targets, row_max, log_sum, log_probs, mean = ctx.saved_tensors
        if not ctx.entropy_requires_grad:
            grad_entropy = None
        grad = logits if ctx.inplace_backward else torch.empty_like(logits)
        _write_logits_grad(
            grad,
            logits,
            rows,
            targets,
            _RowStats(row_max, log_sum, None, mean),
            -torch.expm1(log_probs),  # 1 - p at the target, exact even when p is close to 1
            _contiguous_or_none(grad_log_probs),
            _contiguous_or_none(grad_entropy),
            vocab_start=_vocab_start(logits, ctx.tp_group),
            temperature=ctx.temperature,
        )
        _zero_unscored_rows(grad, rows)
        if ctx.inplace_backward:
            # the kernels write through raw pointers; bump the version so a second reader of these
            # logits (e.g. a retained graph) fails loudly instead of reading the gradient
            torch.autograd.graph.increment_version(logits)
        return grad, None, None, None, None, None, None, None


def _row_statistics(logits, rows, targets, tp_group, temperature, with_entropy) -> _RowStats:
    """Statistics of the selected rows, combined over the vocab-parallel group."""
    vocab_start = _vocab_start(logits, tp_group)
    if logits.is_cuda:
        from miles.backends.training_utils.loss_hub import fused_log_probs_triton  # needs triton

        row_max, row_sum, row_dsum, target = fused_log_probs_triton.row_statistics(
            logits, rows, targets, vocab_start=vocab_start, temperature=temperature, with_entropy=with_entropy
        )
    else:
        row_max, row_sum, row_dsum, target = _row_statistics_torch(
            logits, rows, targets, vocab_start=vocab_start, temperature=temperature, with_entropy=with_entropy
        )
    in_shard = (targets >= vocab_start) & (targets < vocab_start + logits.size(1))
    row_max, row_sum, row_dsum, target = _combine_over_tp(
        row_max, row_sum, row_dsum, target, in_shard, tp_group, temperature
    )
    return _RowStats(row_max, torch.log(row_sum), target, (row_dsum / row_sum if with_entropy else None))


def _combine_over_tp(row_max, row_sum, row_dsum, target, in_shard, tp_group, temperature):
    """Re-measure each shard's statistics from the global max and add them.

    Each shard reports ``sum exp(d)``, ``sum exp(d) * d`` and the target's ``d`` from its own max;
    moving the reference to the global max shifts every ``d`` by the same amount. Only the shard
    holding the target contributes its ``d_y``.
    """
    if tp_group is None or dist.get_world_size(tp_group) == 1:
        return row_max, row_sum, row_dsum, target
    global_max = row_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX, group=tp_group)
    shift = (row_max - global_max) / temperature
    rescale = torch.exp(shift)
    columns = [row_sum * rescale, torch.where(in_shard, target + shift, torch.zeros_like(target))]
    if row_dsum is not None:
        columns.append((row_dsum + shift * row_sum) * rescale)
    sums = torch.stack(columns, dim=1)
    dist.all_reduce(sums, group=tp_group)
    return global_max, sums[:, 0], (sums[:, 2] if row_dsum is not None else None), sums[:, 1]


def _write_logits_grad(
    grad, logits, rows, targets, stats, one_minus_p, grad_log_probs, grad_entropy, *, vocab_start, temperature
):
    if logits.is_cuda:
        from miles.backends.training_utils.loss_hub import fused_log_probs_triton  # needs triton

        fused_log_probs_triton.write_logits_grad(
            grad,
            logits,
            rows,
            targets,
            stats.max,
            stats.log_sum,
            stats.mean,
            one_minus_p,
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
        stats,
        one_minus_p,
        grad_log_probs,
        grad_entropy,
        vocab_start=vocab_start,
        temperature=temperature,
    )


def _zero_unscored_rows(grad, rows):
    if rows.numel() == grad.size(0):  # rows are unique, so every row was scored
        return
    if grad.is_cuda:
        from miles.backends.training_utils.loss_hub import fused_log_probs_triton  # needs triton

        fused_log_probs_triton.zero_unscored_rows(grad, rows)
        return
    unscored = torch.ones(grad.size(0), dtype=torch.bool, device=grad.device)
    unscored[rows] = False
    grad[unscored] = 0


def _row_statistics_torch(logits, rows, targets, *, vocab_start, temperature, with_entropy):
    """This shard's ``(max, sum exp(d), sum exp(d) * d or None, d_y or 0)`` per row, from its own max."""
    x = logits.index_select(0, rows).float()
    row_max = x.max(dim=-1).values if rows.numel() else x.new_empty(0)
    d = (x - row_max.unsqueeze(1)) / temperature
    e = torch.exp(d)
    local_targets = targets - vocab_start
    in_shard = (local_targets >= 0) & (local_targets < logits.size(1))
    target_d = d.gather(1, local_targets.clamp(0, logits.size(1) - 1).unsqueeze(1)).squeeze(1)
    target = torch.where(in_shard, target_d, torch.zeros_like(target_d))
    return row_max, e.sum(dim=-1), ((e * d).sum(dim=-1) if with_entropy else None), target


def _write_logits_grad_torch(
    grad, logits, rows, targets, stats, one_minus_p, grad_log_probs, grad_entropy, *, vocab_start, temperature
):
    d = (logits.index_select(0, rows).float() - stats.max.unsqueeze(1)) / temperature
    p = torch.exp(d - stats.log_sum.unsqueeze(1))
    dd = torch.zeros_like(d)
    if grad_log_probs is not None:
        dd = -grad_log_probs.unsqueeze(1) * p
        local_targets = targets - vocab_start
        owned = ((local_targets >= 0) & (local_targets < logits.size(1))).nonzero().squeeze(1)
        dd[owned, local_targets[owned]] = grad_log_probs[owned] * one_minus_p[owned]
    if grad_entropy is not None:
        dd = dd - grad_entropy.unsqueeze(1) * p * (d - stats.mean.unsqueeze(1))
    grad[rows] = (dd / temperature).to(grad.dtype)


def _vocab_start(logits: Tensor, tp_group: dist.ProcessGroup | None) -> int:
    """First vocab id of this rank's shard: Megatron splits the padded vocab into equal shards."""
    if tp_group is None:
        return 0
    return dist.get_rank(tp_group) * logits.size(1)


def _contiguous_or_none(tensor: Tensor | None) -> Tensor | None:
    return tensor.contiguous() if tensor is not None else None
