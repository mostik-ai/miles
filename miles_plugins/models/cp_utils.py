"""CP token-layout helpers used inside model plugin ``forward`` implementations.

Distinct from ``miles.backends.training_utils.data.context_parallel``, which owns the CP
helpers the training backend applies *around* the model (slicing data, masks,
logprobs and logits). Everything here runs inside a layer.

Megatron CP stores each rank's tokens in the zigzag load-balanced order that
ring attention wants, while fla's CP operators expect a contiguous rank-local
chunk. Anything that hands a sequence to fla under CP therefore has to relayout
first and undo it afterwards.
"""

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from fla.ops.cp import build_cp_context as _fla_build_cp_context
except ImportError:
    _fla_build_cp_context = None


def build_fla_cp_context(cu_seqlens: torch.Tensor, cp_group, conv_kernel_size: int, device: torch.device):
    """fla CP context for a rank of ``cp_group`` from the global packed boundaries ``cu_seqlens``."""
    if _fla_build_cp_context is None:
        raise RuntimeError(
            "Hybrid CP requires fla.ops.cp (flash-linear-attention >= 0.4.2) " "but it could not be imported."
        )
    if cu_seqlens is None or cu_seqlens.numel() < 2:
        raise ValueError(f"Hybrid CP requires valid cu_seqlens (at least 2 elements) but got {cu_seqlens}")
    return _fla_build_cp_context(
        cu_seqlens=cu_seqlens.to(device=device, dtype=torch.int32),
        group=cp_group,
        conv1d_kernel_size=conv_kernel_size,
    )


def build_gdn_cp_context(module: nn.Module, cu_seqlens: torch.Tensor, device: torch.device):
    """Build fla CP context for a GatedDeltaNet module from packed sequence boundaries.

    Args:
        module: GDN module with ``cp_group`` / ``cp_world_size`` / ``conv_kernel_size``.
        cu_seqlens: Global packed sequence boundaries (e.g. ``packed_seq_params.cu_seqlens_q``).
        device: Target device.

    Returns ``None`` when CP is not configured on the module (``cp_group`` not set).
    """
    cp_group = getattr(module, "cp_group", None)
    if cp_group is None:
        return None
    return build_fla_cp_context(cu_seqlens, cp_group, module.conv_kernel_size, device)


def _relayout_indices(cu_seqlens, cp_size, total):
    """Index maps between the two layouts of a ``total``-token packed stream, built on device.

    Rank ``r``'s zigzag shard holds, per sequence, chunks ``r`` and ``2 * cp_size - 1 - r`` of
    ``2 * cp_size`` equal chunks, or its contiguous ``1 / cp_size`` when the length is not a multiple of
    ``2 * cp_size`` (the final padding). Returns ``(zigzag_position, packed_position)`` over the whole
    stream: row ``p`` of the packed stream is row ``zigzag_position[p]`` of the rank-concatenated zigzag
    shards, and ``packed_position`` is the inverse map."""
    cu = cu_seqlens.to(torch.int64)
    lengths = cu[1:] - cu[:-1]
    torch._assert_async(cu[-1] == total)
    torch._assert_async((lengths % cp_size == 0).all())
    shard = total // cp_size
    position = torch.arange(total, device=cu.device)
    seq = torch.searchsorted(cu, position, right=True) - 1
    start = cu[seq]
    length = lengths[seq]
    offset = position - start
    zigzag = length % (2 * cp_size) == 0
    chunk = torch.where(zigzag, length // (2 * cp_size), length // cp_size)
    part = offset // chunk
    mirrored = zigzag & (part >= cp_size)
    owner = torch.where(mirrored, 2 * cp_size - 1 - part, part)
    zigzag_position = owner * shard + start // cp_size + offset % chunk + mirrored * chunk
    packed_position = torch.empty_like(zigzag_position)
    packed_position[zigzag_position] = position
    return zigzag_position, packed_position


class _Exchange:
    """One rank's side of a row permutation across the CP ranks, each holding ``n`` rows: row ``i`` of the result is
    row ``sources[cp_rank * n + i]`` of the ranks' concatenated rows. Each rank sends another only the rows it takes,
    in one all-to-all, instead of every rank gathering every row."""

    def __init__(self, sources, n, cp_rank, cp_size):
        owners = sources // n
        sent = owners == cp_rank
        taken = owners[cp_rank * n : (cp_rank + 1) * n]
        self.send_rows = sources[sent] % n
        self.send_counts = torch.bincount(
            torch.arange(sources.numel(), device=sources.device)[sent] // n, minlength=cp_size
        ).tolist()
        self.receive_counts = torch.bincount(taken, minlength=cp_size).tolist()
        self.order = torch.argsort(taken, stable=True)

    def __call__(self, x, cp_group):
        received = torch.empty_like(x)
        dist.all_to_all_single(
            received, x.index_select(0, self.send_rows), self.receive_counts, self.send_counts, group=cp_group
        )
        rows = torch.empty_like(received)
        rows[self.order] = received
        return rows


def _exchanges(cu_seqlens, cp_rank, cp_size, total):
    """The two directions' exchanges for one packed micro-batch, kept on its ``cu_seqlens`` tensor: every linear
    attention layer, its recompute and its backward relayout the same stream, and building an exchange waits on the
    GPU for its counts. The tensor object, not its storage, carries them, so a later micro-batch never reuses them."""
    key = (cp_rank, cp_size, total)
    cached = getattr(cu_seqlens, "_cp_exchanges", None)
    if cached is None or cached[0] != key:
        to_packed, to_zigzag = _relayout_indices(cu_seqlens, cp_size, total)
        n = total // cp_size
        cached = key, _Exchange(to_packed, n, cp_rank, cp_size), _Exchange(to_zigzag, n, cp_rank, cp_size)
        cu_seqlens._cp_exchanges = cached
    return cached[1], cached[2]


class _Relayout(torch.autograd.Function):
    """A row permutation across CP by all-to-all; the gradient is the inverse permutation."""

    @staticmethod
    def forward(ctx, x, exchange, inverse, cp_group):
        ctx.inverse, ctx.cp_group = inverse, cp_group
        return exchange(x.contiguous(), cp_group)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.inverse(grad_output.contiguous(), ctx.cp_group), None, None, None


def zigzag_to_packed_shard(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    to_packed, to_zigzag = _exchanges(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    return _Relayout.apply(hidden_states, to_packed, to_zigzag, cp_group)


def packed_shard_to_zigzag(hidden_states, cu_seqlens, cp_group, cp_rank, cp_size):
    to_packed, to_zigzag = _exchanges(cu_seqlens, cp_rank, cp_size, hidden_states.size(0) * cp_size)
    return _Relayout.apply(hidden_states, to_zigzag, to_packed, cp_group)
