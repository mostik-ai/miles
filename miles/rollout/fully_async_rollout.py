"""Fully asynchronous rollout generation.

A persistent background worker keeps up to ``rollout_batch_size`` prompt groups in
flight at all times; each training step only drains already-completed groups from the
data buffer (see ``fully_async_data_buffer.py``). Rollout production and training
consumption run in parallel, so per-iteration wall time moves from
``rollout_time + train_time`` toward ``max(rollout_time, train_time)``.

Selected by ``train_async.py --fully-async``, which requires the class-based
rollout API (the default; incompatible with ``MILES_USE_LEGACY_ROLLOUT_V1=1``).

Evaluation targets whatever ``GenerateState`` ``RolloutManager`` passes via
``RolloutFnEvalInput.generate_state`` (see ``miles/rollout/checkpoint_eval.py``
for how the dedicated-fleet state is built). When unset, eval shares the
rollout engines, pausing producer submissions for the duration of the
(blocking) eval.
"""

import asyncio
import logging
from dataclasses import replace

from miles.backends.megatron_utils.megatron_config import resolve_megatron_config
from miles.rollout import async_frontier
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnEvalInput,
    RolloutFnEvalOutput,
    RolloutFnInput,
    RolloutFnOutput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.fully_async_data_buffer import (
    DataBuffer,
    DataBufferConstructorInput,
    DataBufferInput,
    DefaultDataBuffer,
    DefaultMultiDataBuffer,
    Group,
    add_data_buffer_arguments,
    first_sample,
)
from miles.rollout.generate_utils.sample_utils import reward_log_summary, sample_text_preview
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.rollout.inference_rollout.inference_rollout_eval import run_eval_datasets
from miles.rollout.submission_scheduler import make_submission_scheduler
from miles.utils.async_utils import run
from miles.utils.function_registry import load_function
from miles.utils.lifecycle import TrajectoryLifecycle
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

NO_PROGRESS_WARN_SECS = 30.0


class FullyAsyncRolloutFn(BaseRolloutFn):
    """Continuous rollout generation decoupled from training steps.

    The worker runs as a long-lived task on the shared rollout event loop, created
    lazily on the first train call. Which finished groups reach training is the
    data buffer's call (see ``fully_async_data_buffer.py``); this class assembles
    what it hands back into a batch.
    """

    add_arguments = staticmethod(add_data_buffer_arguments)

    def __init__(self, input: RolloutFnConstructorInput):
        super().__init__(input)
        self.args = input.args
        self.data_source = input.data_source
        self.state = GenerateState(input.args)
        # default to sample level backfill for fully async rollout
        self._scheduler = make_submission_scheduler(input.args, default="sample")
        assert input.args.async_unused_samples_handler in ("retry", "drop")
        # applied to every group we do not train on; "drop" discards instead of recycling
        self._handle_unused = (
            self._recycle if input.args.async_unused_samples_handler == "retry" else (lambda prompt_group: None)
        )
        self._sample_filter = load_function(input.args.rollout_sample_filter_path)
        self._worker: asyncio.Task | None = None
        self._eval_prompt_dataset_cache: dict = {}
        self._producer_resumed = asyncio.Event()
        self._producer_resumed.set()
        self._output: DataBuffer | None = None
        # prompt groups submitted and not yet accepted by the buffer: pending snapshots read them
        self._active: dict[asyncio.Task, list[Sample]] = {}
        # completed groups a checkpoint's frontier persisted, queued once the buffer exists
        self._restored: list[DataBufferInput] = []

    async def __call__(self, input: RolloutFnInput) -> RolloutFnOutput:
        if input.evaluation:
            return await self._call_eval(input)
        if self._worker is None:
            default_buffer_cls = (
                DefaultMultiDataBuffer if resolve_megatron_config(self.args).is_multi_policy else DefaultDataBuffer
            )
            buffer_cls = load_function(self.args.custom_async_data_buffer_path) or default_buffer_cls
            self._output = buffer_cls(
                DataBufferConstructorInput(args=self.args, unused_handler_fn=self._handle_unused)
            )
            if self._restored:
                self._output.restore(self._restored)
                self._restored = []
            self._worker = asyncio.create_task(self._worker_loop())
            logger.info("Started fully-async rollout worker")
        return await self._drain(input)

    async def dispose(self) -> None:
        if (worker := self._worker) is None:
            return
        await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_end_worker(worker), worker.get_loop()))

    async def _call_eval(self, input: RolloutFnEvalInput) -> RolloutFnOutput:
        if input.generate_state is not None:
            results = await run_eval_datasets(input.generate_state, self._eval_prompt_dataset_cache)
            return RolloutFnEvalOutput(data=results)

        logger.info("Pausing fully-async producer submissions for shared-engine eval")
        self._producer_resumed.clear()
        try:
            results = await run_eval_datasets(self.state, self._eval_prompt_dataset_cache)
        finally:
            self._producer_resumed.set()
            logger.info("Resumed fully-async producer submissions after eval")
        return RolloutFnEvalOutput(data=results)

    # -------------------------- producer --------------------------

    def _max_in_flight_groups(self) -> int:
        if (x := self.args.async_max_concurrent_samples) is not None:
            # Whole groups are submitted, so the sample budget floors to a group count.
            return max(1, x // self.args.n_samples_per_prompt)
        return self.args.rollout_batch_size

    def _submit_one_group(self) -> tuple[asyncio.Task, list[Sample]]:
        samples = self.data_source.get_samples(1)
        self._scheduler.on_submit(samples)
        [prompt_group] = samples
        return asyncio.create_task(self._generate_group(prompt_group)), prompt_group

    async def _generate_group(self, prompt_group: list[Sample]) -> DataBufferInput:
        result = await generate_and_rm_group(
            self.state,
            prompt_group,
            sampling_params=self.state.sampling_params.copy(),
            evaluation=False,
            sample_done_callback=self._scheduler.sample_done_callback,
        )
        return DataBufferInput(prompt_group=prompt_group, group=result)

    async def _worker_loop(self) -> None:
        active = self._active
        while True:
            await self._producer_resumed.wait()
            while self._scheduler.has_capacity(pending_groups=len(active), group_budget=self._max_in_flight_groups()):
                task, prompt_group = self._submit_one_group()
                active[task] = prompt_group
            done, _ = await self._scheduler.wait_for_progress(set(active))
            for task in done:
                # in flight until the buffer accepts it: a put blocked on a full buffer still owns the group
                await self._output.put(self._collect_group_result(task, active[task]))
                del active[task]

    def _collect_group_result(self, task: asyncio.Task, prompt_group: list[Sample]) -> DataBufferInput:
        if not task.cancelled():
            return task.result()

        logger.warning(
            "Rollout group was cancelled; marking samples aborted: indices=%s",
            [sample.index for sample in prompt_group],
        )
        return DataBufferInput(
            prompt_group=prompt_group,
            group=[replace(sample, status=Sample.Status.ABORTED) for sample in prompt_group],
            cancelled=True,
        )

    def report_pending(self, reason: str, rollout_id: int | None = None) -> None:
        """A work-ledger snapshot of the groups this producer owns that training has not taken: in flight (with the
        response tokens of their finished samples), completed in the buffer, and queued for regeneration.

        Taken at each drain, save and dispose, on the rollout event loop, so it sees one consistent state.
        """
        sink = TrajectoryLifecycle().sink
        if sink is None:
            return
        try:
            active = self._active.values()
            in_flight = {int(group[0].group_index): int(sum(s.response_length for s in group)) for group in active}
            completed = getattr(self._output, "pending", dict)()
            retry = [int(group[0].group_index) for group in getattr(self.data_source, "buffer", [])]
            data = {"rollout_id": rollout_id, "in_flight": in_flight, "completed": completed, "retry": retry}
            tokens = sum(in_flight.values()) + sum(completed.values())
            sink.record("pending", detail=reason, tokens=tokens, data=data)
        except Exception:
            sink.lost += 1  # observability never fails the producer, and a lost record is counted

    def snapshot(self, reason: str, rollout_id: int | None = None) -> None:
        """report_pending from another thread (a save or dispose on the rollout executor)."""

        async def report() -> None:
            self.report_pending(reason, rollout_id)

        try:
            run(report())
        except Exception:
            logger.warning("Pending-work snapshot at %s failed", reason, exc_info=True)

    def save(self, rollout_id: int) -> None:
        """The save's pending snapshot and the rollout frontier, taken in one step of the rollout loop, then written
        beside the data cursor (miles/rollout/async_frontier.py)."""

        async def capture() -> dict:
            self.report_pending("save", rollout_id)
            completed = self._output.completed() if self._output is not None else list(self._restored)
            return async_frontier.capture(self.data_source, completed, list(self._active.values()))

        frontier = run(capture())
        async_frontier.write(frontier, async_frontier.frontier_path(self.args.save, rollout_id))

    def load(self, rollout_id: int | None) -> None:
        """Continue from the frontier saved with the loaded checkpoint, if it has one: its cursor, its prompts first
        in line, and its completed groups back in the buffer with the weight versions they were generated at."""
        frontier = async_frontier.read(self.args.load, rollout_id)
        if frontier is not None:
            self._restored = async_frontier.restore(self.data_source, frontier, self.args.rollout_shuffle)
            logger.info(
                "Restored the rollout frontier of checkpoint %s: %d completed groups, %d to regenerate",
                rollout_id,
                len(self._restored),
                len(frontier["regenerate"]),
            )

    # -------------------------- consumer --------------------------

    async def _next_group(
        self, *, current_version: int | None, trainer_model_id: str | None, rollout_id: int | None = None
    ) -> DataBufferInput:
        queue_get = asyncio.create_task(
            self._output.get(
                current_version=current_version, trainer_model_id=trainer_model_id, rollout_id=rollout_id
            )
        )
        try:
            while True:
                done, _ = await asyncio.wait(
                    {queue_get, self._worker},
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=NO_PROGRESS_WARN_SECS,
                )
                # Checked before the queue: the worker loop never returns normally, so a
                # dead worker fails the step now instead of after its backlog drains.
                if self._worker in done:
                    if self._worker.cancelled():
                        raise RuntimeError("fully-async rollout was disposed while a step waited for groups")
                    self._worker.result()
                    raise RuntimeError("fully-async rollout worker exited without an exception")
                if queue_get in done:
                    return queue_get.result()
                logger.warning(f"No completed rollout groups for {NO_PROGRESS_WARN_SECS}s")
        finally:
            if not queue_get.done():
                queue_get.cancel()

    async def _drain(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        args = self.args
        assert args.rollout_global_dataset

        target_data_size = args.rollout_batch_size
        data: list[Group] = []
        do_print = True

        while len(data) < target_data_size:
            entry = await self._next_group(
                current_version=input.weight_version,
                trainer_model_id=input.trainer_model_id,
                rollout_id=input.rollout_id,
            )
            assert len(entry.group) == args.n_samples_per_prompt

            if do_print:
                sample = first_sample(entry.group)
                logger.info(
                    "First rollout sample: text_preview=%s, label=%s, reward_summary=%s",
                    sample_text_preview(sample),
                    str(sample.label)[:100],
                    reward_log_summary(sample.reward),
                )
                do_print = False

            data.append(entry.group)

        sample = first_sample(data[-1])
        logger.info(
            "Finish rollout: text_preview=%s, label=%s, reward_summary=%s",
            sample_text_preview(sample),
            str(sample.label)[:100],
            reward_log_summary(sample.reward),
        )

        data.sort(key=lambda group: first_sample(group).index)

        if self._sample_filter is not None:
            self._sample_filter(args, data)

        self.report_pending("drain", input.rollout_id)
        return RolloutFnTrainOutput(samples=data, metrics=self._output.get_metrics(input.trainer_model_id))

    def _recycle(self, prompt_group: list[Sample]) -> None:
        for sample in prompt_group:
            sample.reset_for_retry()
        self.data_source.add_samples([prompt_group])


async def _end_worker(worker: asyncio.Task) -> None:
    worker.cancel()
    await asyncio.wait({worker})
