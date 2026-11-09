"""Transformer Engine runs FlashAttention 2 under any context parallelism, keeping FlashAttention 4 off "for context
parallelism yet":
https://github.com/NVIDIA/TransformerEngine/blob/release_v2.17/transformer_engine/pytorch/attention/dot_product_attention/utils.py
Its Ulysses path (cp_comm_type "a2a") moves whole sequences of a share of the heads to each rank before attention, so
its kernel computes plain varlen causal attention, which FlashAttention 4 trains about three times faster on Blackwell.
While that path runs, the FlashAttention 2 varlen kernels it fetches from Transformer Engine's backends module at each
call go to FlashAttention 4; the ring and all-gather paths fetch the same names and keep FlashAttention 2.
"""

import functools
import logging
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch

logger = logging.getLogger(__name__)
_routing = threading.local()
# TODO: delete this module when the image's Transformer Engine wheel is v2.19 or later (NVIDIA/TransformerEngine#3149).


def patch_flash_attn_4_ulysses() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        return
    try:
        from flash_attn.cute import interface
    except ImportError:
        return
    from transformer_engine.pytorch.attention.dot_product_attention import backends, context_parallel

    ulysses = context_parallel.AttnFuncWithCPAndQKVOA2A
    patched = getattr(ulysses, "miles_flash_attn_4", False)  # config-access-exempt: the marker set once patched
    if patched or backends._flash_attn_varlen_fwd is None:
        return
    context_parallel.AttnFuncWithCPAndQKVOA2A = _routing_to_flash_attn_4_while_running(ulysses)
    backends._flash_attn_varlen_fwd = _forward_on_flash_attn_4(backends._flash_attn_varlen_fwd, interface)
    backends._flash_attn_varlen_bwd = _backward_on_flash_attn_4(backends._flash_attn_varlen_bwd, interface)
    logger.info("Transformer Engine's Ulysses attention runs on FlashAttention 4")


def _routing_to_flash_attn_4_while_running(ulysses: type[torch.autograd.Function]) -> type[torch.autograd.Function]:
    class UlyssesOnFlashAttn4(ulysses):
        miles_flash_attn_4 = True

        @staticmethod
        def forward(ctx: Any, *args: Any) -> Any:
            with _routing_to_flash_attn_4():
                return ulysses.forward(ctx, *args)

        @staticmethod
        def backward(ctx: Any, *grad_outputs: Any) -> Any:
            with _routing_to_flash_attn_4():
                return ulysses.backward(ctx, *grad_outputs)

    return UlyssesOnFlashAttn4


@contextmanager
def _routing_to_flash_attn_4() -> Iterator[None]:
    routing = getattr(_routing, "active", False)  # config-access-exempt: each thread starts without the flag
    _routing.active = True
    try:
        yield
    finally:
        _routing.active = routing


def _routes(
    interface: Any, q: torch.Tensor, v: torch.Tensor, max_seqlen_q: int, max_seqlen_k: int, kwargs: dict
) -> bool:
    """FlashAttention 4 takes a call the Ulysses path makes when it is self-attention FlashAttention 4 trains: equal
    query and key lengths, equal query and value heads, no local window or dropout, and head sizes it supports.
    Transformer Engine keeps it off for the hd256 cross-attention and local-window shapes its Blackwell kernels get
    wrong or refuse."""
    active = getattr(_routing, "active", False)  # config-access-exempt: each thread starts without the flag
    if not active or max_seqlen_q != max_seqlen_k or q.shape[-1] != v.shape[-1]:
        return False
    window = (kwargs.get("window_size_left", -1), kwargs.get("window_size_right", -1))
    if window not in ((-1, 0), (-1, -1)) or kwargs.get("dropout_p"):
        return False
    try:
        # the check Transformer Engine runs for FlashAttention 4 outside context parallelism
        interface._validate_head_dims(
            q.shape[-1], v.shape[-1], torch.cuda.get_device_capability()[0], 16 // q.element_size()
        )
    except AssertionError:
        return False
    return True


def _forward_on_flash_attn_4(flash_attn_2_forward: Callable[..., Any], interface: Any) -> Callable[..., Any]:
    @functools.wraps(flash_attn_2_forward)
    def forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs: Any) -> Any:
        if not _routes(interface, q, v, max_seqlen_q, max_seqlen_k, kwargs):
            return flash_attn_2_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs)
        out, lse = interface._flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=kwargs["softmax_scale"],
            causal=kwargs["causal"],
            return_lse=True,
        )[:2]
        # Transformer Engine reads FlashAttention 2's output, log-sum-exp and dropout state at 0, 1 and 3
        return out, lse, None, None

    return forward


def _backward_on_flash_attn_4(flash_attn_2_backward: Callable[..., Any], interface: Any) -> Callable[..., Any]:
    @functools.wraps(flash_attn_2_backward)
    def backward(
        dout, q, k, v, out, lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs: Any
    ) -> Any:
        if not _routes(interface, q, v, max_seqlen_q, max_seqlen_k, kwargs):
            return flash_attn_2_backward(
                dout, q, k, v, out, lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, **kwargs
            )
        # FlashAttention 4 writes the gradients into the dq, dk and dv Transformer Engine passes and reads them from
        return interface._flash_attn_bwd(
            q,
            k,
            v,
            out,
            dout,
            lse,
            softmax_scale=kwargs["softmax_scale"],
            causal=kwargs["causal"],
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            deterministic=kwargs.get("deterministic", False),
            dq=dq,
            dk=dk,
            dv=dv,
        )

    return backward
