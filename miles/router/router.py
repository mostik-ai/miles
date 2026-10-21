import asyncio
import json
import logging
import sys
import time
from collections import deque
from dataclasses import dataclass, field

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.router.config import MilesRouterConfig
from miles.utils.logging_utils import configure_logger_raw
from miles.utils.workers.argv_utils import parse_config_argv

logger = logging.getLogger(__name__)

# Routing prefers these states; a suspect worker takes requests only while none of them is registered.
ROUTABLE_STATES = ("serving", "paused")
HEALTH_CLIENT_MAX_CONNECTIONS = 16
TRANSITION_LOG_LENGTH = 10000


@dataclass
class WorkerState:
    """The router's view of one engine.

    serving and paused take requests; a paused engine is inside an in-place weight update, so its probe timeouts
    and 5xx replies are not failures until ``paused_until``. suspect failed a probe or a request and takes requests
    only while no serving engine is left. failed takes none. recovered is failed but answers probes again: only the
    controller can vouch for its weights, so it stays unrouted until ``/readmit_worker``.
    """

    state: str = "serving"
    since: float = field(default_factory=time.time)
    failures: int = 0
    paused_until: float = 0.0  # a time.monotonic() deadline
    failed_requests: int = 0


def run_router(config: MilesRouterConfig):
    """
    Run the Miles router with the specified configuration.
    """
    # Spawned as a fresh interpreter, so it inherits no logging config.
    configure_logger_raw("miles_router")
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-router")

    # Initialize the router with tokenizer and lazy worker initialization
    miles_router = MilesRouter(config, verbose=False)

    # Start the server
    uvicorn.run(
        miles_router.app, host=config.host, port=config.port, log_level="info", timeout_keep_alive=config.keep_alive
    )


class MilesRouter:
    def __init__(self, config: MilesRouterConfig, verbose=False):
        """Initialize the miles-router with SGLang router address"""
        self.config = config
        self.verbose = verbose

        self.app = FastAPI()
        self.app.router.on_startup.append(self._start_background_health_check)

        # URL -> Active Request Count (load state)
        self.worker_request_counts: dict[str, int] = {}
        self.worker_states: dict[str, WorkerState] = {}
        # Every state change, in order, for the controller to forward as telemetry (GET /worker_states?since=).
        self.transitions: deque[dict] = deque(maxlen=TRANSITION_LOG_LENGTH)
        self._transition_count = 0
        # URL -> upstream requests in flight, cancelled when the worker fails so their clients retry elsewhere
        self._in_flight: dict[str, set[asyncio.Future]] = {}

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=config.max_connections),
            timeout=httpx.Timeout(config.timeout),
        )
        # Probes have their own pool: proxied requests holding every proxy connection starved them into timeouts,
        # which dropped a healthy engine under load.
        self.health_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=HEALTH_CLIENT_MAX_CONNECTIONS),
            timeout=httpx.Timeout(config.health_check_timeout),
        )

        self._setup_routes()

    def _setup_routes(self):
        """Setup all the HTTP routes except catch-all proxy"""
        # sglang-router api
        self.app.post("/add_worker")(self.add_worker)
        self.app.post("/remove_worker")(self.remove_worker)
        self.app.get("/list_workers")(self.list_workers)
        self.app.post("/pause_workers")(self.pause_workers)
        self.app.post("/resume_workers")(self.resume_workers)
        self.app.post("/readmit_worker")(self.readmit_worker)
        self.app.get("/worker_states")(self.get_worker_states)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def _start_background_health_check(self):
        asyncio.create_task(self._health_check_loop())

    async def _probe(self, url: str) -> str:
        """ok, unhealthy, or refused: a refused connection means the engine's process is gone, as no pause does."""
        try:
            response = await self.health_client.get(f"{url}/health", timeout=self.config.health_check_timeout)
        except httpx.ConnectError:
            return "refused"
        except Exception as e:
            logger.debug(f"[miles-router] Worker {url} health check failed: {e}")
            return "unhealthy"
        return "ok" if response.status_code == 200 else "unhealthy"

    async def _health_check_loop(self):
        """Background loop to monitor worker health and adjust routing pool."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                await self._health_check_once()
            except asyncio.CancelledError:
                logger.warning("[miles-router] Background health check loop is being cancelled.")
                raise
            except Exception as e:
                logger.error(f"[miles-router] Unexpected error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def _health_check_once(self) -> None:
        """Probe every registered worker, failed ones included, so that a recovered engine is noticed."""
        probed = list(self.worker_states.items())
        outcomes = await asyncio.gather(*(self._probe(url) for url, _ in probed))
        for (url, worker), outcome in zip(probed, outcomes, strict=True):
            # a result for a url removed or re-registered during the probe describes another registration
            if self.worker_states.get(url) is worker:
                self._observe(url, outcome, "health probe")

    def _observe(self, url: str, outcome: str, source: str) -> None:
        worker = self.worker_states[url]
        if outcome == "ok":
            worker.failures = 0
            if worker.state in ("suspect", "failed"):
                self._transition(url, {"suspect": "serving", "failed": "recovered"}[worker.state], f"{source} passed")
            return
        paused = worker.state == "paused" and time.monotonic() < worker.paused_until
        if worker.state == "recovered":
            self._transition(url, "failed", f"{source} {outcome}")
        elif outcome == "refused":
            self._fail(url, f"{source} refused")
        elif worker.state != "failed" and not paused:
            worker.failures += 1
            if worker.failures >= self.config.health_check_failure_threshold:
                self._fail(url, f"{worker.failures} consecutive failures, the last a {source}")
            elif worker.state == "serving":
                self._transition(url, "suspect", f"{source} {outcome}")

    def _fail(self, url: str, reason: str) -> None:
        if self.worker_states[url].state != "failed":
            self._transition(url, "failed", reason)
        for upstream in self._in_flight.pop(url, set()):
            upstream.cancel()

    def _transition(self, url: str, state: str, reason: str) -> None:
        worker = self.worker_states[url]
        previous, worker.state, worker.since = worker.state, state, time.time()
        self._log_transition(url, previous, state, reason, worker)

    def _log_transition(self, url: str, previous: str, state: str, reason: str, worker: WorkerState) -> None:
        log = logger.warning if state in ("failed", "absent") else logger.info
        log(f"[miles-router] Worker {url}: {previous} -> {state} ({reason})")
        self._transition_count += 1
        self.transitions.append(
            {
                "seq": self._transition_count,
                "ts": time.time(),
                "url": url,
                "from": previous,
                "to": state,
                "reason": reason,
                "failed_requests": worker.failed_requests,
            }
        )

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        result = await self.do_proxy(request, path)
        return self.build_proxy_response(result)

    async def do_proxy(
        self,
        request: Request,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        """Core proxy logic. Returns dict with request_body, response_body, status_code, headers.

        A transport failure answers 502, so the client's retry lands on another worker; a refused connection fails
        the worker at once, and a broken reply counts as one failure.
        """
        worker_url = self._use_url()
        if worker_url is None:
            return self._error_result(503, "no worker is available", body)
        url = f"{worker_url}/{path}"

        if body is None:
            body = await request.body()
        if headers is None:
            headers = dict(request.headers)
        if body is not None:
            headers = {k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding")}

        upstream = asyncio.ensure_future(self._forward(request.method, url, body, headers))
        self._in_flight.setdefault(worker_url, set()).add(upstream)
        try:
            response, content = await upstream
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise
            return self._failed_request(worker_url, "the worker failed while the request was in flight", body)
        except httpx.TransportError as error:
            if worker_url in self.worker_states and isinstance(error, httpx.ConnectError):
                self._observe(worker_url, "refused", "request")
            elif worker_url in self.worker_states and isinstance(error, httpx.ReadError | httpx.RemoteProtocolError):
                self._observe(worker_url, "unhealthy", "request")
            return self._failed_request(worker_url, f"{type(error).__name__}: {error}", body)
        finally:
            self._in_flight.get(worker_url, set()).discard(upstream)
            self._finish_url(worker_url)
        return {
            "request_body": body,
            "response_body": content,
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }

    async def _forward(self, method: str, url: str, body: bytes, headers: dict) -> tuple[httpx.Response, bytes]:
        response = await self.client.request(method, url, content=body, headers=headers)
        return response, await response.aread()

    def _failed_request(self, url: str, message: str, body: bytes | None) -> dict:
        if (worker := self.worker_states.get(url)) is not None:
            worker.failed_requests += 1
        return self._error_result(502, f"worker {url}: {message}", body)

    @staticmethod
    def _error_result(status_code: int, message: str, body: bytes | None) -> dict:
        return {
            "request_body": body,
            "response_body": json.dumps({"error": message}).encode(),
            "status_code": status_code,
            "headers": {"content-type": "application/json"},
        }

    def build_proxy_response(self, result: dict) -> Response:
        """Build HTTP response from proxy result."""
        content = result["response_body"]
        status_code = result["status_code"]
        headers = result["headers"]
        headers = {
            k: v
            for k, v in headers.items()
            if k.lower() not in ("content-length", "transfer-encoding", "server", "date")
        }
        content_type = headers.get("content-type", "")
        try:
            data = json.loads(content)
            return JSONResponse(content=data, status_code=status_code, headers=headers)
        except Exception:
            return Response(content=content, status_code=status_code, headers=headers, media_type=content_type)

    async def add_worker(self, request: Request):
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker  with body {"url": "http://127.0.0.1:10090"}
        """
        worker_url = await self._parse_worker_url(request)
        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        # Add if new, keep a simple request count per worker. A known url keeps its state: a failed worker's
        # weights may be stale, so only /readmit_worker returns it to the pool.
        if worker_url not in self.worker_request_counts:
            self.worker_request_counts[worker_url] = 0
            self.worker_states[worker_url] = worker = WorkerState()
            self._log_transition(worker_url, "absent", "serving", "registered", worker)

        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def remove_worker(self, request: Request):
        """Remove a worker from the router, using the same URL conventions as add_worker."""
        worker_url = await self._parse_worker_url(request)
        if worker_url is None:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )

        self.worker_request_counts.pop(worker_url, None)
        if (worker := self.worker_states.pop(worker_url, None)) is not None:
            self._log_transition(worker_url, worker.state, "absent", "removed", worker)

        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def _parse_worker_url(self, request: Request) -> str | None:
        if worker_url := request.query_params.get("url") or request.query_params.get("worker_url"):
            return worker_url

        body = await request.body()
        payload = json.loads(body) if body else {}
        return payload.get("url") or payload.get("worker_url")

    async def list_workers(self, request: Request):
        """List all registered workers"""
        return {"urls": list(self.worker_request_counts.keys())}

    async def pause_workers(self, request: Request):
        """Workers entering an in-place weight update: probe timeouts and 5xx replies are not failures until the
        pause deadline. A refused connection still is, and a failed worker stays failed."""
        deadline = time.monotonic() + self.config.pause_deadline_seconds
        for url in await self._parse_worker_urls(request):
            if (worker := self.worker_states.get(url)) is not None and worker.state not in ("failed", "recovered"):
                worker.paused_until = deadline
                if worker.state != "paused":
                    self._transition(url, "paused", "weight update")
        return self._states()

    async def resume_workers(self, request: Request):
        for url in await self._parse_worker_urls(request):
            if (worker := self.worker_states.get(url)) is not None and worker.state == "paused":
                self._transition(url, "suspect" if worker.failures else "serving", "weight update ended")
        return self._states()

    async def readmit_worker(self, request: Request):
        """The controller verified this worker's weights: it serves again."""
        worker_url = await self._parse_worker_url(request)
        if worker_url not in self.worker_states:
            return JSONResponse(status_code=404, content={"error": f"worker {worker_url} is not registered"})
        worker = self.worker_states[worker_url]
        if worker.state in ("failed", "recovered"):
            worker.failures = 0
            self._transition(worker_url, "serving", "readmitted")
        return self._states()

    async def get_worker_states(self, request: Request):
        """Each worker's state, and the transitions after sequence number ``since``."""
        since = int(request.query_params.get("since", 0))
        return {
            **self._states(),
            "transitions": [entry for entry in self.transitions if entry["seq"] > since],
            "next": self._transition_count,
        }

    def _states(self) -> dict:
        now = time.monotonic()
        return {
            "workers": {
                url: {
                    "state": worker.state,
                    "since": worker.since,
                    "failures": worker.failures,
                    "paused_seconds_left": max(0.0, worker.paused_until - now) if worker.state == "paused" else 0.0,
                    "failed_requests": worker.failed_requests,
                    "in_flight": self.worker_request_counts.get(url, 0),
                }
                for url, worker in self.worker_states.items()
            }
        }

    async def _parse_worker_urls(self, request: Request) -> list[str]:
        body = await request.body()
        return list(json.loads(body).get("urls", [])) if body else []

    def _use_url(self) -> str | None:
        """Select the worker with minimal active requests, preferring serving and paused workers to suspect ones."""
        states = self.worker_states
        pool = [url for url in self.worker_request_counts if states[url].state in ROUTABLE_STATES] or [
            url for url in self.worker_request_counts if states[url].state == "suspect"
        ]
        if not pool:
            return None
        url = min(pool, key=self.worker_request_counts.get)
        self.worker_request_counts[url] += 1
        return url

    def _finish_url(self, url: str) -> None:
        """Mark the request to the given URL as finished"""
        if (count := self.worker_request_counts.get(url)) is None or count == 0:
            logger.info(f"[miles-router] Request to {url} finished after the worker was deregistered; ignoring")
            return
        self.worker_request_counts[url] = count - 1


if __name__ == "__main__":
    run_router(parse_config_argv(MilesRouterConfig, sys.argv[1:]))
