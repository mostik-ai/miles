"""Opt-in pre-proxy conditioning for TITO sessions.

Some rollout setups need an external artifact to exist, and to be named on the request, *before*
the backend engine sees a model call: a composite whose prompt rows are produced by a second,
frozen model cannot be served correctly if the engine is asked first and conditioned afterwards.
The session server is the only place that sees every call of a session together with its exact
pretokenized `input_ids`, so it is the only place such an artifact can be materialized once per
session and referenced by every later call.

This module is the whole mechanism: a hook resolved from `--session-conditioning-hook-path`
(unset by default, so nothing here runs in an ordinary session), invoked between the request being
prepared and the proxy being made, and the reference it returns carried into the metadata of every
sample assembled from that session.

The hook owns the artifact's identity; the session server only guarantees three things:

* it is called before the backend call, with that call's exact `input_ids`;
* it is given a monotonic per-call sequence number allocated under the session lock, because the
  server releases that lock across the backend call and permits overlapping requests, so nothing
  derived from the committed turn count is unique; and
* every sample assembled from the session carries exactly one reference, and a session that
  changes its reference mid-flight is an error rather than a silently mixed trajectory.
"""

import inspect
from typing import Any, Protocol

from miles.utils.types import Sample

CONDITIONING_METADATA_KEY = "conditioning_ref"


class ConditioningHook(Protocol):
    """`fn(session_id, sequence, input_ids, request_body) -> dict | None`, sync or async.

    The hook may mutate `request_body` in place — setting a namespaced `rid` is the point of
    running before the proxy — and returns the session's conditioning reference, or `None` to
    leave it unchanged.
    """

    def __call__(
        self,
        *,
        session_id: str,
        sequence: int,
        input_ids: list[int],
        request_body: dict[str, Any],
    ) -> dict[str, Any] | None: ...


async def apply_conditioning(
    hook: ConditioningHook,
    *,
    session_id: str,
    sequence: int,
    input_ids: list[int],
    request_body: dict[str, Any],
    previous: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Run the hook before the proxy and return the session's reference."""
    reference = hook(
        session_id=session_id,
        sequence=sequence,
        input_ids=input_ids,
        request_body=request_body,
    )
    if inspect.isawaitable(reference):
        reference = await reference
    if reference is None:
        return previous
    if not isinstance(reference, dict):
        raise TypeError(f"conditioning hook must return a JSON object or None, got {type(reference).__name__}")
    if previous is not None and previous != reference:
        raise ValueError(
            f"session {session_id} changed its conditioning reference during the session "
            f"({previous} -> {reference}); every sample assembled from one session carries one reference"
        )
    return reference


def stamp_conditioning(samples: list[Sample], reference: dict[str, Any] | None) -> None:
    """Carry the session's reference into every sample it produced."""
    if reference is None:
        return
    for sample in samples:
        sample.metadata = {**(sample.metadata or {}), CONDITIONING_METADATA_KEY: reference}
