"""Triton kernels for ``fused_log_probs``: per-row softmax statistics and the logits gradient.

Both kernels read one vocab shard of selected logits rows, in the logits' own dtype, and do all
arithmetic in fp32. To keep the exponential off the critical path they work in base 2
(``y = z * log2(e)``, one ``exp2`` per element), multiply by the reciprocal temperature instead of
dividing, and the statistics kernel rescales its running sums once per block, not per element.
"""

import torch
import triton
import triton.language as tl

# Measured on B300 at [65536, 129280] bf16: the statistics kernel reads at the speed of torch's own
# row reduction (4.2 TB/s) and the gradient kernel moves 6.1 TB/s, 92% of a device copy.
_STATS_BLOCK_V, _STATS_NUM_WARPS = 2048, 2
_GRAD_BLOCK_V, _GRAD_NUM_WARPS = 2048, 4


@triton.jit
def _row_stats_kernel(
    logits_ptr,
    rows_ptr,
    targets_ptr,
    max_ptr,
    sum_ptr,
    zsum_ptr,
    target_logit_ptr,
    stride_row,
    n_vocab,
    vocab_start,
    inv_temperature,
    WITH_ENTROPY: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid = tl.program_id(0)
    row = tl.load(rows_ptr + pid).to(tl.int64)
    row_ptr = logits_ptr + row * stride_row
    lanes = tl.arange(0, BLOCK_V)
    log2_scale = inv_temperature * 1.4426950408889634  # y = z * log2(e), z = logits / T

    # running max m (base 2), sum of 2^(y - m), and sum of 2^(y - m) * y
    run_max = tl.full([], float("-inf"), tl.float32)
    run_sum = tl.zeros([], tl.float32)
    run_ysum = tl.zeros([], tl.float32)
    for start in range(0, n_vocab, BLOCK_V):
        cols = start + lanes
        in_vocab = cols < n_vocab
        y = tl.load(row_ptr + cols, mask=in_vocab, other=float("-inf")).to(tl.float32) * log2_scale
        new_max = tl.maximum(run_max, tl.max(y, axis=0))
        e = tl.exp2(y - new_max)
        rescale = tl.exp2(run_max - new_max)
        run_sum = run_sum * rescale + tl.sum(e, axis=0)
        if WITH_ENTROPY:
            run_ysum = run_ysum * rescale + tl.sum(tl.where(in_vocab, e * y, 0.0), axis=0)
        run_max = new_max

    # back to natural units: m = y_max * ln 2, sum exp(z - m) = run_sum, sum exp(z - m) * z = ln 2 * run_ysum
    tl.store(max_ptr + pid, run_max * 0.6931471805599453)
    tl.store(sum_ptr + pid, run_sum)
    if WITH_ENTROPY:
        tl.store(zsum_ptr + pid, run_ysum * 0.6931471805599453)

    target = tl.load(targets_ptr + pid) - vocab_start
    in_shard = (target >= 0) & (target < n_vocab)
    target_z = tl.load(row_ptr + target, mask=in_shard, other=0.0).to(tl.float32) * inv_temperature
    tl.store(target_logit_ptr + pid, tl.where(in_shard, target_z, 0.0))


@triton.jit
def _logits_grad_kernel(
    logits_ptr,
    grad_ptr,
    rows_ptr,
    targets_ptr,
    lse_ptr,
    mu_ptr,
    grad_log_probs_ptr,
    grad_entropy_ptr,
    stride_row,
    grad_stride_row,
    n_vocab,
    vocab_start,
    inv_temperature,
    HAS_GRAD_LOG_PROBS: tl.constexpr,
    HAS_GRAD_ENTROPY: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    pid_row = tl.program_id(0)
    pid_block = tl.program_id(1)
    row = tl.load(rows_ptr + pid_row).to(tl.int64)
    cols = pid_block * BLOCK_V + tl.arange(0, BLOCK_V)
    in_vocab = cols < n_vocab

    z = tl.load(logits_ptr + row * stride_row + cols, mask=in_vocab, other=0.0).to(tl.float32) * inv_temperature
    p = tl.exp2((z - tl.load(lse_ptr + pid_row)) * 1.4426950408889634)
    dz = tl.zeros([BLOCK_V], tl.float32)
    if HAS_GRAD_LOG_PROBS:
        g = tl.load(grad_log_probs_ptr + pid_row)
        target = tl.load(targets_ptr + pid_row) - vocab_start
        dz = tl.where(cols == target, g, 0.0) - g * p
    if HAS_GRAD_ENTROPY:
        c = tl.load(grad_entropy_ptr + pid_row)
        dz = dz - c * p * (z - tl.load(mu_ptr + pid_row))
    grad = dz * inv_temperature
    tl.store(grad_ptr + row * grad_stride_row + cols, grad.to(grad_ptr.dtype.element_ty), mask=in_vocab)


def row_statistics(logits, rows, targets, *, vocab_start: int, temperature: float, with_entropy: bool):
    """This shard's ``(max, sum exp(z - max), sum exp(z - max) * z or None, target z or 0)`` per row."""
    n_rows = rows.numel()
    stats = torch.empty((4, n_rows), dtype=torch.float32, device=logits.device)
    if n_rows:
        _row_stats_kernel[(n_rows,)](
            logits,
            rows,
            targets,
            stats[0],
            stats[1],
            stats[2],
            stats[3],
            logits.stride(0),
            logits.size(1),
            vocab_start,
            1.0 / temperature,
            WITH_ENTROPY=with_entropy,
            BLOCK_V=_STATS_BLOCK_V,
            num_warps=_STATS_NUM_WARPS,
        )
    row_max, row_sum, row_zsum, target_logit = stats
    return row_max, row_sum, (row_zsum if with_entropy else None), target_logit


def write_logits_grad(
    grad, logits, rows, targets, lse, mu, grad_log_probs, grad_entropy, *, vocab_start: int, temperature: float
):
    """Write the gradient of the selected rows into ``grad``; ``grad`` may be ``logits`` itself."""
    n_rows = rows.numel()
    if not n_rows:
        return
    grid = (n_rows, triton.cdiv(logits.size(1), _GRAD_BLOCK_V))
    _logits_grad_kernel[grid](
        logits,
        grad,
        rows,
        targets,
        lse,
        mu if mu is not None else lse,
        grad_log_probs if grad_log_probs is not None else lse,
        grad_entropy if grad_entropy is not None else lse,
        logits.stride(0),
        grad.stride(0),
        logits.size(1),
        vocab_start,
        1.0 / temperature,
        HAS_GRAD_LOG_PROBS=grad_log_probs is not None,
        HAS_GRAD_ENTROPY=grad_entropy is not None,
        BLOCK_V=_GRAD_BLOCK_V,
        num_warps=_GRAD_NUM_WARPS,
    )
