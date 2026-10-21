"""The rollout frontier a checkpoint persists beside its data cursor: the groups a fully-async producer had drawn and
training had not taken when the checkpoint was saved.

The data cursor counts groups drawn, not trained, so a resume that restores only the cursor skips every group that
was in flight, buffered or queued for regeneration at the save. The frontier holds the cursor too, captured with the
groups in one step of the rollout event loop: completed groups whole (samples, rewards, weight-version spans, routing
arrays), and the prompts of groups still generating or queued for regeneration. A resume queues the completed groups
back in the data buffer, where the stale filter judges them by the versions they were generated at, and puts the
prompts first in line, so the next drains take exactly the frontier's groups before any new draw.
"""

import copy
import os
import pickle

from miles.rollout.fully_async_data_buffer import DataBufferInput

# The data source's cursor fields (RolloutDataSource.save writes the same ones).
CURSOR = ("epoch_id", "sample_group_index", "sample_index", "sample_offset", "metadata")


def frontier_path(root: str, rollout_id: int) -> str:
    return os.path.join(root, f"rollout/async_frontier_{rollout_id}.pt")


def detached(item, memo: dict):
    """A copy of a sample (or nested list of them) the live producer cannot change while it is written: a sample's
    fields are rebound, never mutated in place, except its metadata. Shared samples stay shared through `memo`."""
    if isinstance(item, list):
        return [detached(value, memo) for value in item]
    if id(item) not in memo:
        twin = copy.copy(item)
        twin.metadata = copy.deepcopy(item.metadata)
        memo[id(item)] = twin
    return memo[id(item)]


def prompts(group: list) -> list:
    """A group's prompts, reset for regeneration; each sample keeps its group index and index."""
    fresh = detached(group, {})
    for sample in fresh:
        sample.reset_for_retry()
    return fresh


def capture(source, completed: list[DataBufferInput], active: list[list]) -> dict:
    """The frontier at this instant; call it on the rollout event loop, so no producer step runs in between."""
    memo = {}
    return {
        "cursor": {name: copy.deepcopy(getattr(source, name)) for name in CURSOR if hasattr(source, name)},
        "completed": [
            DataBufferInput(
                prompt_group=detached(entry.prompt_group, memo),
                group=detached(entry.group, memo),
                cancelled=entry.cancelled,
            )
            for entry in completed
        ],
        "regenerate": [prompts(group) for group in [*active, *getattr(source, "buffer", [])]],
    }


def write(frontier: dict, destination: str) -> None:
    """Streamed to a partial file and renamed into place, so a reader never sees part of one."""
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    partial = f"{destination}.partial"
    with open(partial, "wb") as stream:
        pickle.dump(frontier, stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, destination)


def restore(source, frontier: dict, shuffle: bool) -> list[DataBufferInput]:
    """Set the data source to the frontier's cursor and queue its prompts first; the completed groups to buffer."""
    for name, value in frontier["cursor"].items():
        setattr(source, name, value)
    if shuffle and getattr(source, "dataset", None) is not None:
        source.dataset.shuffle(source.epoch_id)
    source.buffer[:0] = frontier["regenerate"]
    return frontier["completed"]


def read(root: str | None, rollout_id: int | None) -> dict | None:
    """The frontier saved at `rollout_id` under `root`, or None where there is none (a checkpoint saved without one)."""
    if root is None or rollout_id is None or not os.path.isfile(path := frontier_path(root, rollout_id)):
        return None
    with open(path, "rb") as stream:
        return pickle.load(stream)
