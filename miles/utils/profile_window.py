"""A profile of chosen microbatches of one update, on chosen training ranks.

`--profile-window-rollout k` opens the window at training forward microbatch a of update k and closes it at forward
microbatch b, or at the update's end (`--profile-window-microbatches a b`), on the ranks `--profile-window-ranks`
names (every rank when unset). Under the 1F1B schedule the window holds those forwards and the backwards interleaved
with them. Inside it Megatron's NVTX ranges are on and the CUDA profiler API capture is open, so an attached Nsight
Systems (`--capture-range=cudaProfilerApi`) records only the window. With `--profile-window-torch`, torch.profiler
records CPU and CUDA activity with shapes, one profiler step per microbatch, and writes one Chrome trace per rank to
`--tensorboard-dir`; stacks and memory are opt-in (`--profile-window-stack`, `--profile-window-memory`).
"""

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def set_nvtx(enabled: bool) -> None:
    from megatron.core.utils import configure_nvtx_profiling

    configure_nvtx_profiling(enabled)


def set_capture(enabled: bool) -> None:
    if torch.cuda.is_available():
        runtime = torch.cuda.cudart()
        torch.cuda.check_error(runtime.cudaProfilerStart() if enabled else runtime.cudaProfilerStop())


class ProfileWindow:
    def __init__(self):
        self.args = None
        self.rank = None
        self.microbatch = None
        self.profiler = None

    def configure(self, args, rank: int) -> None:
        start, end = args.profile_window_microbatches
        if args.profile_window_rollout is not None and not 0 <= start < end:
            raise ValueError(f"--profile-window-microbatches needs 0 <= start < end, got {start} {end}")
        self.args, self.rank = args, rank

    def begin_update(self, rollout_id: int) -> None:
        args = self.args
        chosen = (
            args is not None
            and args.profile_window_rollout == rollout_id
            and (not args.profile_window_ranks or self.rank in args.profile_window_ranks)
        )
        self.microbatch = 0 if chosen else None

    def step(self) -> None:
        """Mark the start of a training forward microbatch."""
        if self.microbatch is None:
            return
        start, end = self.args.profile_window_microbatches
        if self.microbatch == start:
            self._open(end - start)
        elif start < self.microbatch < end and self.profiler is not None:
            self.profiler.step()
        elif self.microbatch == end:  # stopping the profiler ends the window's last step
            self._close()
        self.microbatch += 1

    def end_update(self) -> None:
        """Close a window the update ended inside."""
        start, end = self.args.profile_window_microbatches if self.args is not None else (0, 0)
        if self.microbatch is not None and start < self.microbatch <= end:
            self._close()
        self.microbatch = None

    def _open(self, steps: int) -> None:
        args = self.args
        logger.info(f"Profile window opens: update {args.profile_window_rollout}, rank {self.rank}")
        set_nvtx(True)
        set_capture(True)
        if args.profile_window_torch:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            self.profiler = torch.profiler.profile(
                activities=activities,
                schedule=torch.profiler.schedule(wait=0, warmup=0, active=steps, repeat=1),
                on_trace_ready=self._write,
                record_shapes=True,
                with_stack=args.profile_window_stack,
                profile_memory=args.profile_window_memory,
            )
            self.profiler.start()

    def _close(self) -> None:
        if self.profiler is not None:
            self.profiler.stop()
            self.profiler = None
        set_capture(False)
        set_nvtx(False)
        logger.info(f"Profile window closes: update {self.args.profile_window_rollout}, rank {self.rank}")

    def _write(self, profiler) -> None:
        directory = Path(self.args.tensorboard_dir)
        directory.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(directory / f"rollout{self.args.profile_window_rollout}_rank{self.rank}.json"))


WINDOW = ProfileWindow()
