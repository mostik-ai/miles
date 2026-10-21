"""Per-rank GPU memory by owner at fixed points, for a ledger that must conserve every byte (Coral's memory ledger).

With `--memory-ledger-dir`, each training rank appends one JSON line to `rank<rank>.jsonl` there: after the model and
optimizer are built, at the end of each update (before the replay queues are cleared and the peaks reset), and after
each save. A line holds the device's total and free bytes, the caching allocator's active and reserved bytes with
their peaks since the previous update ended, and the bytes each owner holds on the device: the DDP parameter and
gradient buffers, the optimizer's master weights (its parameter groups) and state tensors (its moments), routing-replay
staging, snapshot buffers, and DeepEP's NVLink buffer, which lives outside the allocator. Each storage counts once,
for the first owner holding it. Host-side replay staging is reported apart. A failed measurement writes its error
instead of partial numbers, so the ledger reads it as missing; it never stops training.
"""

import json
import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def storage_bytes(tensors, device_type: str, seen: set) -> int:
    total = 0
    for tensor in tensors:
        if not torch.is_tensor(tensor) or tensor.device.type != device_type:
            continue
        storage = tensor.untyped_storage()
        if storage.data_ptr() and storage.data_ptr() not in seen:
            seen.add(storage.data_ptr())
            total += storage.nbytes()
    return total


def inner_optimizers(optimizer) -> list:
    parts = getattr(optimizer, "chained_optimizers", None) or ([] if optimizer is None else [optimizer])
    return [getattr(part, "optimizer", part) for part in parts]


def owners(model_chunks, optimizer, replays=(), snapshots=(), device_type: str = "cuda") -> dict[str, int]:
    """Bytes per owner on one device type; a storage two owners share counts for the first."""
    seen: set = set()
    buffers = [
        buffer
        for chunk in model_chunks
        for buffer in (*getattr(chunk, "buffers", ()), *getattr(chunk, "expert_parallel_buffers", ()))
    ]
    inner = inner_optimizers(optimizer)
    parameters = [tensor for b in buffers for tensor in ([b.param_data] if b.param_data is not None else b.params)]
    return {
        "parameters": storage_bytes(parameters, device_type, seen),
        "gradients": storage_bytes((b.grad_data for b in buffers), device_type, seen),
        "master": storage_bytes((p for o in inner for group in o.param_groups for p in group["params"]), device_type, seen),
        "moments": storage_bytes((v for o in inner for state in o.state.values() for v in state.values()), device_type, seen),
        "replay": storage_bytes((t for replay in replays for t in replay.top_indices_list), device_type, seen),
        "snapshots": storage_bytes(snapshots, device_type, seen),
    }


def collective_bytes() -> int:
    from megatron.core.transformer.moe import fused_a2a

    buffer = getattr(fused_a2a, "_buffer", None)
    return 0 if buffer is None else int(buffer.num_nvl_bytes)


def measure(model_chunks, optimizer, replays) -> dict:
    stats = torch.cuda.memory_stats()
    free, total = torch.cuda.mem_get_info()
    return {
        "device": {"total": total, "free": free},
        "allocator": {
            "active": stats.get("active_bytes.all.current"),
            "active_peak": stats.get("active_bytes.all.peak"),
            "reserved": stats.get("reserved_bytes.all.current"),
            "reserved_peak": stats.get("reserved_bytes.all.peak"),
            "retries": stats.get("num_alloc_retries"),
        },
        "owners": {**owners(model_chunks, optimizer, replays), "collective": collective_bytes()},
        "host": {"replay": sum(owners([], None, replays, device_type="cpu").values())},
    }


def write(args, point: str, rollout_id: int, *, model_chunks, optimizer, replays=()) -> None:
    if (directory := getattr(args, "memory_ledger_dir", None)) is None:
        return
    rank = torch.distributed.get_rank()
    record = {"point": point, "rollout_id": rollout_id, "rank": rank, "time": time.time()}
    try:
        from miles.utils.misc import get_current_node_ip

        record |= {"node": get_current_node_ip(), "gpu": torch.cuda.current_device()}
        record |= measure(model_chunks, optimizer, replays)
    except Exception as error:  # measurement must never stop training; the ledger reads this record as missing
        logger.warning(f"Memory ledger measurement failed at {point}: {error!r}")
        record["error"] = repr(error)
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(Path(directory) / f"rank{rank:03d}.jsonl", "a") as stream:
            stream.write(json.dumps(record) + "\n")
    except OSError as error:
        logger.warning(f"Memory ledger write failed at {point}: {error!r}")
