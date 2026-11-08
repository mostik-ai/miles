import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

_LOGGED_SAMPLES = 3


@dataclass(frozen=True)
class ExpertParallelLoad:
    """The layer and expert-parallel rank that receive the most routed copies of one sample's tokens."""

    num_tokens: int
    layer: int
    ep_rank: int
    copies: int
    share: float
    tokens_without_routing: int


def expert_parallel_load(routed_experts: torch.Tensor, num_experts: int, ep_size: int) -> ExpertParallelLoad:
    """Find the busiest layer and expert-parallel rank of one sample's replayed routing.

    Args:
        routed_experts: `[num_tokens, num_layers, topk]` expert ids from the rollout; -1 marks no expert.
            Expert `e` lives on rank `e // (num_experts // ep_size)`, as Megatron lays out local experts.
    """
    num_tokens, num_layers, _ = routed_experts.shape
    no_expert = ep_size
    ranks = torch.where(routed_experts < 0, no_expert, routed_experts // (num_experts // ep_size))
    layer_offsets = torch.arange(num_layers, dtype=ranks.dtype).view(1, num_layers, 1) * (ep_size + 1)
    counts = torch.bincount((ranks + layer_offsets).flatten(), minlength=num_layers * (ep_size + 1))
    routed = counts.view(num_layers, ep_size + 1)[:, :no_expert]
    layer, ep_rank = divmod(int(routed.argmax()), ep_size)
    copies = int(routed[layer, ep_rank])
    return ExpertParallelLoad(
        num_tokens=num_tokens,
        layer=layer,
        ep_rank=ep_rank,
        copies=copies,
        share=copies / max(int(routed[layer].sum()), 1),
        tokens_without_routing=int((routed_experts < 0).all(dim=-1).any(dim=-1).sum()),
    )


def log_expert_parallel_load(
    rollout_id: int, routed_experts: list[torch.Tensor], num_experts: int, ep_size: int
) -> None:
    """Log the samples whose replayed routing sends the most copies to a single expert-parallel rank."""
    loads = sorted(
        (expert_parallel_load(sample, num_experts, ep_size) for sample in routed_experts),
        key=lambda load: load.copies,
        reverse=True,
    )
    busiest = "; ".join(
        f"{load.num_tokens} tokens: EP rank {load.ep_rank} gets {load.share:.0%} of layer {load.layer}'s copies "
        f"({load.copies}, {load.share * ep_size:.1f}x an even share)"
        for load in loads[:_LOGGED_SAMPLES]
    )
    tokens_without_routing = sum(load.tokens_without_routing for load in loads)
    logger.info(
        f"rollout {rollout_id}: routing replay load over {len(loads)} samples, busiest: {busiest}; "
        f"{tokens_without_routing} tokens without routing"
    )
