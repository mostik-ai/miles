from __future__ import annotations

from pathlib import Path

CHECKPOINT_TRACKER_FILENAME = "latest_checkpointed_iteration.txt"
LORA_ADAPTER_DIRNAME = "adapter"


def read_checkpoint_tracker_iteration(checkpoint_root: str | Path | None) -> int | None:
    if checkpoint_root is None:
        return None

    tracker = Path(checkpoint_root) / CHECKPOINT_TRACKER_FILENAME
    if not tracker.is_file():
        return None

    content = tracker.read_text().strip()
    if not content.isdigit():
        return None
    return int(content)


def write_checkpoint_tracker_iteration(checkpoint_root: str | Path, iteration: int) -> None:
    """Name ``iteration`` the newest checkpoint under ``checkpoint_root``; a reader never sees a half-written tracker."""
    tracker = Path(checkpoint_root) / CHECKPOINT_TRACKER_FILENAME
    written = tracker.with_suffix(".writing")
    written.write_text(str(iteration))
    written.replace(tracker)


def lora_adapter_checkpoint_dir(checkpoint_root: str | Path, iteration: int) -> Path:
    return Path(checkpoint_root) / f"iter_{iteration:07d}" / LORA_ADAPTER_DIRNAME


def tracked_lora_adapter_dir(checkpoint_root: str | Path | None) -> Path | None:
    """The adapter directory of the iteration the tracker names; None without a tracker, or when that iteration
    is not a LoRA save."""
    iteration = read_checkpoint_tracker_iteration(checkpoint_root)
    if iteration is None:
        return None
    adapter_dir = lora_adapter_checkpoint_dir(checkpoint_root, iteration)
    return adapter_dir if adapter_dir.is_dir() else None
