"""Engine admission: an engine takes requests only once it proves it holds the current weights.

The proof is the engine's reported weight version and its per-tensor checksum map, judged against a reference:

1. an expected map another component supplies for the version (the parent checkpoint's map at a first sync, or an
   evaluation snapshot's receipt), when there is one;
2. otherwise the map the trusted engines (those already serving the version) agree on;
3. otherwise, with no trusted engine, the map every candidate agrees on.

Without a reference nobody is admitted: two engines that disagree cannot tell which of them is wrong. An engine that
started from dummy weights is also refused while any tensor still holds its dummy checksum: engines that start alike
agree on a tensor the update never wrote.
This module also speaks the Miles router's engine-state API (pause, resume, readmit, states).
"""

import asyncio
from collections.abc import Sequence

import httpx

from miles.utils.audit_utils.checksum_utils import InferenceEngineChecksums, flatten_inference_engine_checksums

VERIFIED = "verified"
STALE_VERSION = "stale_version"
CHECKSUM_MISMATCH = "checksum_mismatch"
UNREACHABLE = "unreachable"  # the connection was refused: the engine's process is gone
UNANSWERED = "unanswered"  # a timeout or an error reply: no proof either way
DUMMY_WEIGHTS = "dummy_weights"  # a tensor still holds the value the engine's dummy load gave it

ROUTER_TIMEOUT_SECONDS = 30.0
PROBE_TIMEOUT_SECONDS = 5.0
MODEL_INFO_TIMEOUT_SECONDS = 5.0
CHECKSUM_TIMEOUT_SECONDS = 30.0
CHECKSUM_REQUEST = {"action": "checksum", "allow_quant_error": False, "selector": "all"}

Observation = tuple[str, InferenceEngineChecksums] | BaseException


async def _router(client: httpx.AsyncClient, router_url: str, method: str, path: str, **kwargs) -> dict:
    response = await client.request(method, f"{router_url}{path}", timeout=ROUTER_TIMEOUT_SECONDS, **kwargs)
    response.raise_for_status()
    return response.json()


async def pause_workers(client: httpx.AsyncClient, router_url: str, urls: Sequence[str]) -> dict:
    return await _router(client, router_url, "POST", "/pause_workers", json={"urls": list(urls)})


async def resume_workers(client: httpx.AsyncClient, router_url: str, urls: Sequence[str]) -> dict:
    return await _router(client, router_url, "POST", "/resume_workers", json={"urls": list(urls)})


async def readmit_worker(client: httpx.AsyncClient, router_url: str, url: str) -> dict:
    return await _router(client, router_url, "POST", "/readmit_worker", json={"url": url})


async def remove_worker(client: httpx.AsyncClient, router_url: str, url: str) -> dict:
    return await _router(client, router_url, "POST", "/remove_worker", json={"url": url})


async def worker_states(client: httpx.AsyncClient, router_url: str, since: int = 0) -> dict:
    """{"workers": {url: {"state", ...}}, "transitions": [changes after sequence number since], "next": int}."""
    return await _router(client, router_url, "GET", "/worker_states", params={"since": since})


async def probe(client: httpx.AsyncClient, url: str) -> str:
    """ok, unhealthy, or refused: only a refused connection proves the engine's process is gone."""
    try:
        response = await client.get(f"{url}/health", timeout=PROBE_TIMEOUT_SECONDS)
    except httpx.ConnectError:
        return "refused"
    except httpx.HTTPError:
        return "unhealthy"
    return "ok" if response.status_code == 200 else "unhealthy"


async def engine_weights(client: httpx.AsyncClient, url: str) -> tuple[str, InferenceEngineChecksums]:
    """The weight version an engine serves and its checksum map, flattened across its ranks."""
    info = await client.get(f"{url}/model_info", timeout=MODEL_INFO_TIMEOUT_SECONDS)
    info.raise_for_status()
    body = await client.post(f"{url}/weights_checker", json=CHECKSUM_REQUEST, timeout=CHECKSUM_TIMEOUT_SECONDS)
    body.raise_for_status()
    return str(info.json()["weight_version"]), flatten_inference_engine_checksums([body.json()])[0]


async def observe(client: httpx.AsyncClient, urls: Sequence[str]) -> dict[str, Observation]:
    """Each engine's version and map, or the exception that kept us from reading them."""
    results = await asyncio.gather(*(engine_weights(client, url) for url in urls), return_exceptions=True)
    return dict(zip(urls, results, strict=True))


def agreed(maps: Sequence[InferenceEngineChecksums]) -> InferenceEngineChecksums | None:
    """The one map all given maps equal, or None if there are none or they differ."""
    return maps[0] if maps and all(checksums == maps[0] for checksums in maps) else None


def reference_checksums(
    observed: dict[str, Observation],
    version: str | None,
    trusted: Sequence[str],
    expected: InferenceEngineChecksums | None = None,
) -> InferenceEngineChecksums | None:
    if expected is not None:
        return expected
    current = {url: result[1] for url, result in observed.items() if _current(result, version)}
    held = [current[url] for url in trusted if url in current]
    return agreed(held) if held else agreed(list(current.values()))


def judge(
    observed: dict[str, Observation],
    version: str | None,
    trusted: Sequence[str],
    expected: InferenceEngineChecksums | None = None,
    dummy: dict[str, InferenceEngineChecksums] | None = None,
) -> dict[str, str]:
    """A verdict per observed engine; a None version accepts whatever version the engine reports. ``dummy`` maps an
    engine that started from dummy weights to the checksums it held then."""
    reference = reference_checksums(observed, version, trusted, expected)
    verdicts = {}
    for url, result in observed.items():
        if isinstance(result, BaseException):
            verdicts[url] = UNREACHABLE if isinstance(result, httpx.ConnectError) else UNANSWERED
        elif not _current(result, version):
            verdicts[url] = STALE_VERSION
        elif any(result[1].get(name) == held for name, held in (dummy or {}).get(url, {}).items()):
            verdicts[url] = DUMMY_WEIGHTS
        else:
            verdicts[url] = VERIFIED if reference is not None and result[1] == reference else CHECKSUM_MISMATCH
    return verdicts


def _current(result: Observation, version: str | None) -> bool:
    return not isinstance(result, BaseException) and (version is None or result[0] == version)
