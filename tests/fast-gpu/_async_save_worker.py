"""One-rank worker: Miles' save_checkpoint schedules an async save on Megatron's persistent worker after SGLang has
patched torch's tensor reducers, as a trainer's first save does after its weight updates."""

import argparse
import os
from argparse import Namespace

import torch
import torch.distributed as dist
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.dist_checkpointing.strategies.torch import TorchDistSaveShardedStrategy, ipc_staging_scope
from megatron.training import global_vars
from megatron.training.async_utils import maybe_finalize_async_save, schedule_async_save

from miles.backends.megatron_utils import checkpoint
from miles.backends.megatron_utils.sglang import monkey_patch_torch_reductions


def _sharded(tensors: dict[str, torch.Tensor]) -> dict[str, ShardedTensor]:
    return {name: ShardedTensor.from_rank_offsets(name, tensor, replica_id=0) for name, tensor in tensors.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True)
    directory = parser.parse_args().directory
    saved = os.path.join(directory, "checkpoint")
    os.makedirs(saved)

    torch.cuda.set_device(0)
    dist.init_process_group("cpu:gloo,cuda:nccl", init_method=f"file://{directory}/rendezvous", rank=0, world_size=1)
    global_vars.set_args(
        Namespace(
            async_save=True,
            async_strategy="nvrx",
            use_persistent_ckpt_worker=True,
            async_ckpt_use_cpu_shm=True,
            async_ckpt_cpu_priority=10,
            async_ckpt_io_priority=None,
        )
    )
    monkey_patch_torch_reductions()
    tensors = {f"weight{i}": torch.randn(1024, 1024, device="cuda") for i in range(4)}

    def save_like_megatron() -> None:
        # what Megatron's save_checkpoint does for an async torch_dist save, with tensors for the model and optimizer
        request = dist_checkpointing.save(
            _sharded(tensors),
            saved,
            TorchDistSaveShardedStrategy(cpu_shm_mode=True),
            async_sharded_save=True,
            async_strategy="nvrx",
        )
        with ipc_staging_scope():
            schedule_async_save(request)

    checkpoint._save_checkpoint_megatron = save_like_megatron
    checkpoint.save_checkpoint()
    maybe_finalize_async_save(blocking=True, terminate=True)

    loaded = dist_checkpointing.load(
        _sharded({name: torch.empty_like(tensor) for name, tensor in tensors.items()}), saved
    )
    assert all(torch.equal(loaded[name].cuda(), tensor) for name, tensor in tensors.items())
    print("PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
