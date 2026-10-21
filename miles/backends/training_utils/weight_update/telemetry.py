"""Where a weight sync's time goes on the rank that drives it, and what it moved.

Global rank 0 drives the engine session and, under a gathered placement, is the only sender. It closes one
dashboard phase per stage: each bucket's export (time blocked in the iterator: the lockstep bridge gather and
conversion) and send (RPC, broadcast and both engines' acks), and the session's begin, drain and connect. It also
counts the object collectives the export makes, each a host sync. Its totals reach every rank over gloo once per
sync, so the tracking rank logs them as one ``refresh/*`` row, keyed by the weight version the sync installed.
"""

import time
from contextlib import contextmanager

import torch
import torch.distributed as dist

from miles.utils.timer import Timer
from miles.utils.tracking_utils import tracking

TOTALS = (
    "started_at",
    "buckets",
    "tensors",
    "bytes",
    "export_seconds",
    "send_seconds",
    "object_collectives",
    "object_collective_seconds",
)
OBJECT_COLLECTIVES = ("all_gather_object", "broadcast_object_list")


def emit_phase(name: str, t0: float, t1: float) -> None:
    """A closed dashboard phase with explicit times: no open event and no perf time, unlike Timer."""
    for sink in Timer().event_sinks:
        sink(name, t0, t1)


@contextmanager
def driver_phase(name: str):
    """A phase closed on global rank 0 only, the rank whose timeline the refresh breakdown reads."""
    t0 = time.time()
    try:
        yield
    finally:
        if dist.get_rank() == 0:
            emit_phase(name, t0, time.time())


class RefreshMeter:
    """One sync's stage timings and totals, measured on the driving rank; a no-op elsewhere."""

    def __init__(self, measuring: bool) -> None:
        self.measuring = measuring
        self.totals = dict.fromkeys(TOTALS, 0.0)
        self.totals["started_at"] = time.time()

    def exported(self, buckets):
        """The iterator's buckets, closing an export phase for the time blocked in each ``next``, the last included."""
        iterator = iter(buckets)
        while True:
            t0 = time.time()
            bucket = next(iterator, None)
            self._close("update_weights_export", "export_seconds", t0)
            if bucket is None:
                return
            yield bucket

    @contextmanager
    def sending(self, bucket):
        """Time one bucket's send; its sizes are read first, since a protocol may clear the bucket."""
        if self.measuring:
            self.totals["buckets"] += 1
            self.totals["tensors"] += len(bucket)
            self.totals["bytes"] += sum(tensor.numel() * tensor.element_size() for _, tensor in bucket)
        t0 = time.time()
        try:
            yield
        finally:
            self._close("update_weights_send", "send_seconds", t0)

    @contextmanager
    def phase(self, name: str):
        t0 = time.time()
        try:
            yield
        finally:
            self._close(name, None, t0)

    @contextmanager
    def counting_object_collectives(self):
        """Count and time torch.distributed's object collectives while the export runs (Megatron-Bridge calls them
        through the module attribute, so a temporary wrapper sees every call)."""
        originals = {name: getattr(dist, name) for name in OBJECT_COLLECTIVES} if self.measuring else {}

        def counted(function):
            def call(*args, **kwargs):
                t0 = time.time()
                try:
                    return function(*args, **kwargs)
                finally:
                    self.totals["object_collectives"] += 1
                    self.totals["object_collective_seconds"] += time.time() - t0

            return call

        for name, function in originals.items():
            setattr(dist, name, counted(function))
        try:
            yield
        finally:
            for name, function in originals.items():
                setattr(dist, name, function)

    def _close(self, name: str, total: str | None, t0: float) -> None:
        if not self.measuring:
            return
        t1 = time.time()
        if total is not None:
            self.totals[total] += t1 - t0
        emit_phase(name, t0, t1)

    def share(self, group) -> dict[str, float]:
        """The driving rank's totals, on every rank: one broadcast from global rank 0."""
        values = torch.tensor([self.totals[name] for name in TOTALS], dtype=torch.float64)
        dist.broadcast(values, src=0, group=group)
        return dict(zip(TOTALS, values.tolist(), strict=True))


def log_refresh(args, totals: dict[str, float], weight_version: int) -> None:
    """One row per sync on the tracking rank, beside the rollout and train rows."""
    row = {f"refresh/{name}": value for name, value in totals.items()}
    tracking.log(args, {**row, "refresh/step": weight_version}, step_key="refresh/step")
