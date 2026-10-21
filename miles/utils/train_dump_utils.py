import logging
from pathlib import Path

import torch

from miles.backends.training_utils.parallel import get_parallel_state

logger = logging.getLogger(__name__)


def save_debug_train_data(args, *, rollout_id, rollout_data):
    if args.save_debug_train_data is not None:
        parallel_state = get_parallel_state()
        # TP peers duplicate the DP shard and only the last PP stage computes the per-token fields
        if parallel_state.tp.rank != 0 or not parallel_state.is_pp_last_stage:
            return
        save_debug_train_data_for_rank(
            args,
            rollout_id=rollout_id,
            rollout_data=rollout_data,
            rank=torch.distributed.get_rank(),
            cp_rank=parallel_state.cp.rank,
            cp_size=parallel_state.cp.size,
        )


def save_debug_train_data_for_rank(args, *, rollout_id, rollout_data, rank, cp_rank=0, cp_size=1):
    """Write one rank's slice of a rollout.

    ``cp_rank`` / ``cp_size`` are passed in for the same reason ``rank`` is:
    this runs outside the process group in tests and offline tools.
    """
    if (path_template := args.save_debug_train_data) is not None:
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                rollout_data=rollout_data,
                cp_rank=cp_rank,
                cp_size=cp_size,
                qkv_format=args.qkv_format,
            ),
            path,
        )


def save_routing_overlap(args, *, rollout_id, rollout_data, log):
    """One file per rank per update: every rank holds its own layers (PP) and its own sequence slice (SP)."""
    if args.dump_details is None or not log:
        return
    parallel_state = get_parallel_state()
    layers = sorted({layer for layer, _, _ in log})
    assert len(log) % len(layers) == 0, f"{len(log)} overlap records for {len(layers)} layers"
    # Each microbatch's forward visits this stage's MoE layers once, in order.
    microbatches = [log[start : start + len(layers)] for start in range(0, len(log), len(layers))]
    path = Path(args.dump_details) / "routing_overlap" / f"{rollout_id}_{torch.distributed.get_rank()}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            rollout_id=rollout_id,
            dp_rank=parallel_state.intra_dp.rank,
            tp_rank=parallel_state.tp.rank,
            tp_size=parallel_state.tp.size,
            pp_rank=parallel_state.pp.rank,
            layers=layers,
            # identifies the DP shard and its packing, so a reader needs no rank-order assumption
            sample_indices=list(rollout_data["sample_indices"]),
            micro_batch_indices=rollout_data["micro_batch_indices"],
            total_lengths=list(rollout_data["total_lengths"]),
            # [tokens in this rank's sequence slice, local MoE layers]; 255 marks padding
            overlap=[torch.stack([o for _, o, _ in mb], dim=1).cpu() for mb in microbatches],
            mass=[torch.stack([m for _, _, m in mb], dim=1).cpu() for mb in microbatches],
        ),
        path,
    )
