import asyncio
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.dashboard import hooks as dashboard_hooks
from miles.ray.rollout import engine_admission as admission
from miles.ray.rollout.eval_fleet import EvalFleetInfo, EvalFleetPin, InferenceControllerEvalFleet
from miles.ray.rollout.rollout_server import RolloutServer, create_rollout_servers
from miles.ray.rollout.router_manager import resolve_router_addrs
from miles.ray.rollout.server_cell import ServerCell, ServerCellMetadata
from miles.utils import async_utils
from miles.utils.audit_utils.checksum_utils import InferenceEngineChecksums
from miles.utils.audit_utils.event_logger.logger import get_event_logger, is_event_logger_initialized
from miles.utils.audit_utils.event_logger.models import InferenceEngineWeightChecksumEvent
from miles.utils.audit_utils.process_identity import SimpleProcessIdentity
from miles.utils.context_lock import (
    ContextLock,
    acquires_lock,
    enforce_lock_discipline,
    lock_exempt,
    releases_lock,
    requires_lock,
    with_lock,
)
from miles.utils.ft_utils.api_server.models import CellStatus
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.init_once import InitOnce, init_once
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import SimpleTicker
from miles.utils.test_utils.fault_injector import FailureMode
from miles.utils.workers.registration.hub import RegistrationHub
from miles.utils.workers.registration.models import RegistrationSnapshot
from miles.utils.workers.worker_provider.base import BaseWorkerProvider, CellInfo, StopWatchFn
from miles.utils.workers.worker_provider.utils import apply_cell_observation

logger = logging.getLogger(__name__)

TICK_INTERVAL_SECONDS = 5.0
RESTART_BACKOFF_SECONDS = 5.0
RESTART_BACKOFF_MAX_SECONDS = 300.0
CELL_TICK_TIMEOUT_SECONDS = 120.0
CELLS_READY_POLL_INTERVAL_SECONDS = 2.0
CELLS_READY_TIMEOUT_SECONDS = 3600.0


@enforce_lock_discipline
class InferenceController:
    @lock_exempt
    def __init__(
        self,
        args,
        *,
        engine_provider: BaseWorkerProvider,
        router_providers: Sequence[BaseWorkerProvider],
    ) -> None:
        self._init_once = InitOnce(type(self).__name__)
        self.args = args
        self._engine_provider = engine_provider
        self._router_providers = router_providers
        self.context_lock = ContextLock("InferenceController")
        self.servers: dict[str, RolloutServer] = {}
        self._eval_fleet: InferenceControllerEvalFleet | None = None
        self._watcher_disposers: list[StopWatchFn] = []
        self._ticker: SimpleTicker | None = None
        # The weight version the serving engines were last updated to, and per version an explicit reference map
        # that replaces agreement between engines: the parent checkpoint's map at a first sync, or an evaluation
        # snapshot's receipt. Engine lifecycle needs the Miles router, whose engine states it drives.
        self.serving_version: str | None = None
        self.expected_checksums: dict[str, InferenceEngineChecksums] = {}
        # The map the first completed update must produce (--expected-initial-weight-checksums), whatever its version:
        # a failed transfer still advances the trainer's version counter.
        self.initial_checksums = _initial_checksums(args.expected_initial_weight_checksums)
        self._router_cursor = 0
        self._http = GeneralHttpClientProvider.client
        self._restarts: dict[str, tuple[int, float]] = {}  # cell id -> (attempts, earliest next attempt)

    @lock_exempt
    @init_once
    async def init(self) -> None:
        configure_logger(self.args, source=SimpleProcessIdentity(component="inference_controller"))

        if not self.args.starts_inference_engines:
            return

        await self._engine_provider.init()
        router_addrs = await resolve_router_addrs(self.args, router_providers=self._router_providers)
        self.servers = await create_rollout_servers(
            self.args,
            context_lock=self.context_lock,
            engine_provider=self._engine_provider,
            router_addrs=router_addrs,
        )
        if self.args.eval_num_gpus > 0:
            self._eval_fleet = InferenceControllerEvalFleet(self.args, srv=self.servers["eval"])

        self._watcher_disposers.append(await self._engine_provider.watch_cells(self._reconcile))
        self._ticker = SimpleTicker(self._tick_cells, interval_seconds=TICK_INTERVAL_SECONDS)

        dashboard_hooks.register_router(self.args)

        await self.wait_expected_num_cells()

    # TEMPORARY: exists only so a suspend can take this lock, reverted with the weight-update fault tolerance work
    @with_lock
    async def stop_cell_between_weight_updates(self, cell_id: str) -> None:
        await self._engine_provider.stop_cells(cell_ids=[cell_id])

    # TEMPORARY: exists only so fault injection can take this lock, reverted with the weight-update fault tolerance work
    @with_lock
    async def inject_fault_between_weight_updates(self, cell_id: str, *, mode: FailureMode, sub_index: int) -> None:
        # TEMPORARY: colocate cannot kill rollout workers while trainer ranks own the shared GPUs
        server = next((srv for srv in self.servers.values() if cell_id in srv.server_cells), None)
        if server is None:
            raise KeyError(f"Unknown rollout cell {cell_id!r}")
        if not server.health_checker_activeness.get().active:
            raise RuntimeError(f"Rollout cell {cell_id!r} is offloaded; refusing fault injection")

        await self._engine_provider._worker_manager_handle.inject_fault.remote(
            cell_id,
            mode=mode.value,
            worker_in_cell_index=sub_index,
        )

    # -------------------------- take over -----------------------------

    @lock_exempt
    async def is_initialized(self) -> bool:
        return self._init_once.is_initialized()

    @lock_exempt
    async def wait_expected_num_cells(self, timeout: float = CELLS_READY_TIMEOUT_SECONDS) -> None:
        await asyncio.gather(*[srv.wait_init_expected_num_cells(timeout=timeout) for srv in self.servers.values()])

    @with_lock
    async def abort_all(self) -> None:
        await async_utils.gather_and_raise_first([srv.abort_all() for srv in self.servers.values()])

    # -------------------------- registration -----------------------------

    @lock_exempt
    async def registration_ingest(self, *, snapshot: RegistrationSnapshot) -> None:
        assert self.servers, (
            f"this inference controller is not ready: reporter {snapshot.reporter_id} announced its cells before "
            f"the orchestration script built this controller's servers, so it does not know yet which models this "
            f"run serves; the reporter announces them again, and they are taken in once the controller is up"
        )
        assert isinstance(provider := self._engine_provider, RegistrationHub), (
            f"run {self.args.run_uuid} deploys its own engines and takes no registration "
            f"(reporter {snapshot.reporter_id})"
        )
        for cell in snapshot.cells:
            assert (
                model_id := cell.info.meta.get("model_id")
            ) in self.servers, (
                f"cell {cell.info.cell_id} serves model {model_id!r}, this run serves {sorted(self.servers)}"
            )
        await provider.ingest(snapshot)

    # -------------------------- rollout lifecycle hooks -----------------------------

    @with_lock
    async def prepare_rollout(self, rollout_id: int, model_id: str | None = None) -> None:
        await self._health_monitoring_resume(model_id)
        await dashboard_hooks.register_engines(self.servers, provider=self._engine_provider)

    @with_lock
    async def prepare_eval(self, model_id: str | None = None) -> None:
        await self._health_monitoring_resume(model_id)
        await dashboard_hooks.register_engines(self.servers, provider=self._engine_provider)

    @with_lock
    async def dispose(self) -> None:
        if (ticker := self._ticker) is not None:
            self._ticker = None
            await ticker.dispose()

        for disposer in self._watcher_disposers:
            await disposer()
        self._watcher_disposers = []

        for srv in self.servers.values():
            await srv.dispose()

    # -------------------------- offload/onload -----------------------------

    # TODO may parallelly execute offload/onload across services
    @with_lock
    async def offload(self, tags: list[str] | None = None) -> None:
        await self._offload(tags=tags)

    @with_lock
    async def onload(self, tags: list[str] | None = None) -> None:
        await self._onload(tags=tags)

    @with_lock
    async def onload_weights(self) -> None:
        if "weight" not in self.args.offload_rollout_level:
            return
        await self._onload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    @with_lock
    async def onload_kv(self) -> None:
        await self._onload(tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_CUDA_GRAPH])

    @with_lock
    async def offload_kv(self) -> None:
        tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
        if "kv_cache" in self.args.offload_rollout_level:
            tags.append(GPU_MEMORY_TYPE_KV_CACHE)
        await self._offload(tags=tags)

    @with_lock
    async def offload_weights(self) -> None:
        if "weight" not in self.args.offload_rollout_level:
            return
        await self._offload(tags=[GPU_MEMORY_TYPE_WEIGHTS])

    @requires_lock
    async def _offload(self, tags: list[str] | None):
        await self._health_monitoring_pause(None)
        for srv in self.servers.values():
            await srv.offload(tags=tags)

    @requires_lock
    async def _onload(self, tags: list[str] | None):
        for srv in self.servers.values():
            await srv.onload(tags)

    # -------------------------- engine management -----------------------------

    @acquires_lock
    async def start_update_weights(self, model_id: str | None = None) -> "UpdatableEngines":
        """Return engines eligible for weight updates."""
        await self._health_monitoring_pause(model_id)
        await self._ensure_cells_ready(model_id=model_id)

        srv = self._get_updatable_server(model_id=model_id)
        if not srv:
            return UpdatableEngines(
                rollout_engines=[],
                engine_gpu_counts=[],
                engine_gpu_offsets=[],
                snapshot_cell_id_to_hashes={},
            )
        if self.args.use_miles_router:
            await self._fail_dead_cells(srv)
            urls = [cell.server_url for cell in srv.weight_update_cells]
            await admission.pause_workers(self._http(), srv.router_url, urls)

        return UpdatableEngines(
            rollout_engines=srv.api_clients,
            engine_gpu_counts=srv.engine_gpu_counts,
            engine_gpu_offsets=srv.engine_gpu_offsets,
            snapshot_cell_id_to_hashes={cell.meta.cell_id: cell.meta.workers_hash for cell in srv.weight_update_cells},
        )

    @releases_lock
    async def abort_update_weights(self) -> None:
        pass

    @releases_lock
    async def end_update_weights(
        self,
        snapshot_cell_id_to_hashes: dict[str, str],
        *,
        completed: bool = True,
        version: int | None = None,
        model_id: str | None = None,
    ) -> list[InferenceEngineChecksums] | None:
        """Admit the updated engines whose weights verify, and return their checksum maps when that took a check.

        An update that did not complete admits nothing. Engines entering service (pending weights, or failed by the
        router) are verified before they take requests; engines already serving are left to ``audit_weights``, off
        the training loop. Without a version (a skipped update) pending engines are admitted unverified, as before.
        The first completed update is judged against the initial map when one is given, and fails the job when it
        leaves no engine serving: with admission closed nothing can generate the batch a next update needs.
        """
        srv = self._get_updatable_server(model_id=model_id)
        if srv is None:
            return None
        cells = [
            cell
            for cell in srv.weight_update_cells
            if snapshot_cell_id_to_hashes.get(cell.meta.cell_id) == cell.meta.workers_hash
        ]
        if self.args.use_miles_router:
            await admission.resume_workers(self._http(), srv.router_url, [cell.server_url for cell in cells])
        if not completed:
            return None
        if version is None or not self.args.use_miles_router:
            await asyncio.gather(*[cell.mark_weights_ready() for cell in cells if cell.is_pending_weights])
            return None
        first, self.serving_version = self.serving_version is None, str(version)
        if first and self.initial_checksums is not None:
            self.expected_checksums.setdefault(self.serving_version, self.initial_checksums)
        states = await self._router_states(srv)
        entering = [c.server_url for c in cells if c.is_pending_weights or _router_failed(states, c.server_url)]
        checksums = await self._verify(srv, cells, entering) if entering else None
        if first and not any(cell.is_serving for cell in srv.weight_update_cells):
            raise RuntimeError(f"No engine's weights verified after the first weight update (version {version})")
        return checksums

    @with_lock
    async def audit_weights(self, version: int | None, model_id: str | None = None) -> list[InferenceEngineChecksums]:
        """The checksum maps of the addressable engines, read after an update that admitted no engine.

        A dead engine is failed, and one at a stale version, or off an explicit reference, is demoted. Agreement
        alone demotes nobody: two engines that disagree cannot tell which of them is wrong.
        """
        srv = self._get_updatable_server(model_id=model_id)
        if srv is None:
            return []
        cells = [cell for cell in srv.weight_update_cells if cell.is_pending_weights_or_serving]
        if self.args.use_miles_router and version is not None and str(version) == self.serving_version:
            return await self._verify(srv, cells, entering=[])
        observed = await admission.observe(self._http(), [cell.server_url for cell in cells])
        return [result[1] for result in observed.values() if not isinstance(result, BaseException)]

    @requires_lock
    async def _verify(self, srv: RolloutServer, cells: list[ServerCell], entering: list[str]) -> list:
        urls = [cell.server_url for cell in cells]
        observed = await admission.observe(self._http(), urls)
        expected = self.expected_checksums.get(self.serving_version)
        trusted = [url for url in urls if url not in entering]
        dummy = {cell.server_url: cell.dummy_checksums for cell in cells if cell.dummy_checksums is not None}
        verdicts = admission.judge(observed, self.serving_version, trusted, expected, dummy)
        if {admission.CHECKSUM_MISMATCH, admission.DUMMY_WEIGHTS} & set(verdicts.values()):
            logger.error(f"Engine weights disagree at version {self.serving_version}: {verdicts}")
        for cell in cells:
            await self._settle(srv, cell, verdicts[cell.server_url], cell.server_url in entering, expected is not None)
        return [result[1] for result in observed.values() if not isinstance(result, BaseException)]

    @requires_lock
    async def _settle(
        self, srv: RolloutServer, cell: ServerCell, verdict: str, entering: bool, explicit: bool
    ) -> None:
        """Apply one verdict. A serving engine is demoted for a stale version, and for a mismatch or a tensor left
        at its dummy value when it enters service or against an explicit reference; an unanswered check changes
        nothing."""
        if verdict == admission.UNREACHABLE:
            await cell.mark_failed()
        elif verdict == admission.VERIFIED and entering and cell.is_pending_weights:
            await cell.mark_weights_ready()
            # a replacement at a url the router still holds as failed is registered already: readmit it
            await admission.readmit_worker(self._http(), srv.router_url, cell.server_url)
        elif verdict == admission.VERIFIED and entering:
            await admission.readmit_worker(self._http(), srv.router_url, cell.server_url)
        elif cell.is_serving and verdict in (
            admission.STALE_VERSION, admission.CHECKSUM_MISMATCH, admission.DUMMY_WEIGHTS
        ):
            if entering or explicit or verdict == admission.STALE_VERSION:
                await cell.demote()

    @requires_lock
    async def _fail_dead_cells(self, srv: RolloutServer) -> None:
        """A refused connection proves an engine dead; so does any failed probe of one the router already failed."""
        cells = srv.weight_update_cells
        states = await self._router_states(srv)
        outcomes = await asyncio.gather(*(admission.probe(self._http(), cell.server_url) for cell in cells))
        for cell, outcome in zip(cells, outcomes, strict=True):
            if outcome == "refused" or (outcome != "ok" and _router_failed(states, cell.server_url)):
                logger.warning(f"Engine {cell.server_url} is dead ({outcome}); leaving it out of weight updates")
                await cell.mark_failed()

    @requires_lock
    async def _readmit_recovered(self) -> None:
        """Between updates, verify the serving engines the router failed that answer again, with the lock released,
        and readmit those still at the current version once it is taken again."""
        srv, version = self._get_updatable_server(), self.serving_version
        if srv is None or version is None:
            return
        states = await self._router_states(srv)
        cells = [cell for cell in srv.weight_update_cells if cell.is_serving]
        recovered = [cell for cell in cells if states.get(cell.server_url, {}).get("state") == "recovered"]
        if not recovered:
            return
        trusted = [c.server_url for c in cells if states.get(c.server_url, {}).get("state") in ("serving", "paused")]
        async with self.context_lock.with_released():
            observed = await admission.observe(self._http(), [cell.server_url for cell in recovered] + trusted)
        if self.serving_version != version:
            return  # an update ran meanwhile, and its end verified these engines
        expected = self.expected_checksums.get(version)
        verdicts = admission.judge(observed, version, trusted, expected)
        for cell in recovered:
            if cell.is_serving:
                await self._settle(srv, cell, verdicts[cell.server_url], True, expected is not None)
        # the maps that vouched for it, as an update's admission logs its own
        maps = [result[1] for result in observed.values() if not isinstance(result, BaseException)]
        if maps and is_event_logger_initialized():
            # no rollout drove this readmission, so it carries the stamp the out-of-loop startup sync carries
            get_event_logger().log(
                InferenceEngineWeightChecksumEvent,
                dict(rollout_id=self.args.start_rollout_id - 1, engine_checksums=maps),
            )

    @requires_lock
    async def _router_states(self, srv: RolloutServer) -> dict[str, dict]:
        """The router's engine states; the changes since the last read go to the dashboard with their own times."""
        report = await admission.worker_states(self._http(), srv.router_url, since=self._router_cursor)
        self._router_cursor = report["next"]
        dashboard_hooks.report_engine_states(report["transitions"])
        return report["workers"]

    @requires_lock
    async def _ensure_cells_ready(self, model_id: str | None = None) -> None:
        deadline = time.monotonic() + CELLS_READY_TIMEOUT_SECONDS
        while True:
            cells = [cell for srv in self._get_servers_of_model_id(model_id) for cell in srv.server_cells.values()]
            if self.args.colocate:
                await asyncio.gather(*[cell.init() for cell in cells if cell.is_uninitialized])
            pending = [cell for cell in cells if not (cell.is_pending_weights_or_serving or cell.is_failed)]
            # a replacement that is still starting joins a later update instead of holding this one
            ready = any(cell.is_pending_weights_or_serving for cell in cells)
            restarting = self.args.rollout_restart_failed_engines and ready
            if not pending or restarting:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {CELLS_READY_TIMEOUT_SECONDS}s waiting for "
                    f"{len(pending)}/{len(cells)} cells to become ready"
                )
            logger.info(f"Waiting for {len(pending)}/{len(cells)} cells to become ready...")
            async with self.context_lock.with_released():
                await asyncio.sleep(CELLS_READY_POLL_INTERVAL_SECONDS)

    @requires_lock
    def _get_servers_of_model_id(self, model_id: str | None) -> list[RolloutServer]:
        if model_id is None:
            return list(self.servers.values())
        srv = self.servers.get(model_id)
        assert srv is not None, f"No server for model_id {model_id!r}, known ids: {sorted(self.servers)}"
        return [srv]

    @requires_lock
    def _get_updatable_server(self, model_id: str | None = None) -> RolloutServer | None:
        if model_id is not None:
            [srv] = self._get_servers_of_model_id(model_id)
            assert srv.update_weights, f"Server for model_id {model_id!r} is frozen (update_weights=False)"
            return srv

        updatable = [srv for srv in self.servers.values() if srv.update_weights]
        match updatable:
            case []:
                return None
            case [srv]:
                return srv
            case _:
                raise ValueError(
                    f"Multiple servers have update_weights=True: {[srv.model_name for srv in updatable]}. "
                    f"Pass model_id to update exactly one of them."
                )

    # -------------------------- eval fleet -----------------------------

    @lock_exempt
    async def get_eval_fleet_info(self) -> EvalFleetInfo | None:
        return self._eval_fleet.info if self._eval_fleet is not None else None

    @lock_exempt
    async def pin_eval_fleet(self, checkpoint_dir: str, weight_version: str) -> EvalFleetPin:
        assert (
            self._eval_fleet is not None
        ), "this run deploys no eval fleet, so nothing asked this controller to pin one"
        return await self._eval_fleet.pin(checkpoint_dir=checkpoint_dir, weight_version=weight_version)

    # -------------------------- misc APIs -----------------------------

    @lock_exempt
    async def get_cell_statuses(self) -> dict[str, CellStatus]:
        return {
            cell_id: cell.cell_status()
            for srv in list(self.servers.values())
            for cell_id, cell in list(srv.server_cells.items())
        }

    @with_lock
    async def check_weights(
        self,
        action: str,
        allow_quant_error: bool = False,
        selector: str = "all",
        skip_list: list[str] | None = None,
        model_id: str | None = None,
    ) -> list[Any]:
        # Only the updatable model is re-synced; a frozen model would always mismatch.
        srv = self._get_updatable_server(model_id=model_id)
        if srv is None:
            return []
        return await srv.check_weights(
            action=action, allow_quant_error=allow_quant_error, selector=selector, skip_list=skip_list
        )

    # -------------------------- tick -----------------------------

    @with_lock
    async def _tick_cells(self) -> None:
        cells = [cell for srv in list(self.servers.values()) for cell in list(srv.server_cells.values())]
        results = await asyncio.gather(
            *[asyncio.wait_for(cell.tick(), timeout=CELL_TICK_TIMEOUT_SECONDS) for cell in cells],
            return_exceptions=True,
        )
        for cell, result in zip(cells, results, strict=True):
            if isinstance(result, BaseException):
                logger.error(f"Ticking cell {cell.meta.cell_id} failed", exc_info=result)
        if self.args.use_miles_router:
            await self._readmit_recovered()
        if self.args.rollout_restart_failed_engines:
            await self._restart_failed()

    @requires_lock
    async def _restart_failed(self) -> None:
        """Relaunch failed engines, backing off 5 * 2^n s (at most 300 s) between attempts on one cell. The watch
        replaces the cell, and the new engine serves only after a weight update verifies it."""
        now = time.monotonic()
        for cell_id, cell in [item for srv in self.servers.values() for item in srv.server_cells.items()]:
            if cell.is_serving:
                self._restarts.pop(cell_id, None)
            attempts, due = self._restarts.get(cell_id, (0, now))
            if not cell.is_failed or now < due:
                continue
            delay = min(RESTART_BACKOFF_SECONDS * 2 ** (attempts + 1), RESTART_BACKOFF_MAX_SECONDS)
            self._restarts[cell_id] = (attempts + 1, now + delay)
            logger.warning(f"Restarting failed engine cell {cell_id} (attempt {attempts + 1})")
            await self._engine_provider.stop_cells(cell_ids=[cell_id])
            await self._engine_provider.start_cells(cell_ids=[cell_id])

    # -------------------------- reconcile -----------------------------

    @with_lock
    async def _reconcile(self, cell_id: str, observed: CellInfo | None) -> None:
        actual_srv: RolloutServer | None = None
        actual_cell: ServerCell | None = None
        for srv in self.servers.values():
            if (c := srv.server_cells.get(cell_id)) is not None:
                actual_srv, actual_cell = srv, c
                break

        async def _add(_cell_id: str, observed_info: CellInfo) -> None:
            observed_cell_meta = _compute_server_cell_meta_from_info(observed_info)
            await self.servers[observed_cell_meta.model_id].add_cell(observed_cell_meta)

        async def _remove(remove_cell_id: str) -> None:
            await actual_srv.remove_cell(remove_cell_id)

        await apply_cell_observation(
            cell_id=cell_id,
            observed=observed,
            actual_workers_hash=actual_cell.meta.workers_hash if actual_cell is not None else None,
            add=_add,
            remove=_remove,
        )

    # -------------------------- utils -----------------------------

    @requires_lock
    async def _health_monitoring_pause(self, model_id: str | None) -> None:
        for srv in self._get_servers_of_model_id(model_id):
            srv.health_checker_activeness.bump_active(False)

    @requires_lock
    async def _health_monitoring_resume(self, model_id: str | None) -> None:
        for srv in self._get_servers_of_model_id(model_id):
            srv.health_checker_activeness.bump_active(True)


def _initial_checksums(path: str | None) -> InferenceEngineChecksums | None:
    if path is None:
        return None
    checksums = json.loads(Path(path).read_text())
    if not isinstance(checksums, dict) or not checksums or not all(isinstance(v, str) for v in checksums.values()):
        raise ValueError(f"{path} is not a checksum map of 'rank<r>/<tensor>' names")
    return checksums


def _router_failed(states: dict[str, dict], url: str) -> bool:
    return states.get(url, {}).get("state") in ("failed", "recovered")


@dataclass(frozen=True)
class UpdatableEngines:
    rollout_engines: list[SGLangApiClient]
    engine_gpu_counts: list[int]
    engine_gpu_offsets: list[int]
    snapshot_cell_id_to_hashes: dict[str, str]


# TODO may move and generalize later
def _compute_server_cell_meta_from_info(info: CellInfo) -> ServerCellMetadata:
    return ServerCellMetadata(
        model_id=info.meta["model_id"],
        worker_type=info.meta["worker_type"],
        cell_id=info.cell_id,
        num_gpus_per_engine=info.meta["num_gpus_per_engine"],
        gpu_offset=info.meta["gpu_offset"],
        sglang_api_key=info.meta["sglang_api_key"],
        worker_name=info.worker_names[0],
        needs_offload=info.meta["needs_offload"],
        update_weights=info.meta["update_weights"],
        workers_hash=info.workers_hash,
    )
