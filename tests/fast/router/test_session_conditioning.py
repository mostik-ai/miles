"""The opt-in pre-proxy conditioning hook (`--session-conditioning-hook-path`).

Drives `SessionCore.chat_completions` in-process against a recording backend, the
`test_session_samples_op.py` precedent: the point of the hook is that the artifact exists and is
named on the request *before* the backend is called, so the tests assert on what the backend
received, not only on what the hook returned.
"""

from __future__ import annotations

import asyncio
import json
import uuid

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config

from miles.rollout.session.conditioning import CONDITIONING_METADATA_KEY, apply_conditioning, stamp_conditioning
from miles.rollout.session.core import SessionCore
from miles.rollout.session.linear_trajectory import SessionRegistry
from miles.rollout.session.samples.codec import decode_samples_and_merge_input_sample
from miles.utils.chat_template_utils import get_tito_tokenizer
from miles.utils.function_registry import function_registry
from miles.utils.processing_utils import load_tokenizer
from miles.utils.types import Sample

HF_CHECKPOINT = "Qwen/Qwen3-0.6B"


class RecordingBackend:
    """A backend that answers every chat call and remembers the bodies it was given."""

    def __init__(self) -> None:
        self.bodies: list[dict] = []

    async def do_proxy(self, request, path, *, body, headers):
        self.bodies.append(json.loads(body))
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                    "meta_info": {"output_token_logprobs": [[-0.5, 3866]], "completion_tokens": 1},
                }
            ]
        }
        return {
            "status_code": 200,
            "response_body": json.dumps(response).encode(),
            "headers": {"content-type": "application/json"},
        }


def build_core(backend, hook_path: str | None = None) -> SessionCore:
    config = make_session_server_config(
        hf_checkpoint=HF_CHECKPOINT,
        apply_chat_template_kwargs={"enable_thinking": False},
        instance_id=uuid.uuid4().hex,
        session_conditioning_hook_path=hook_path,
    )
    tokenizer = load_tokenizer(config.hf_checkpoint, chat_template_path=None, trust_remote_code=True)
    tito_tokenizer = get_tito_tokenizer(
        tokenizer, tokenizer_type=config.tito_model, chat_template_kwargs=config.apply_chat_template_kwargs
    )
    registry = SessionRegistry(tokenizer, tito_tokenizer=tito_tokenizer)
    return SessionCore(backend, registry, config, config.instance_id)


def chat(core, session_id: str, messages: list[dict]) -> None:
    asyncio.run(
        core.chat_completions(
            session_id,
            method="POST",
            query="",
            headers={},
            body=json.dumps({"messages": messages, "model": "m"}).encode(),
        )
    )


def collect(core, session_id: str) -> list[Sample]:
    response = asyncio.run(core.collect_samples(session_id, max_seq_len=None))
    assert response.status_code == 200, response.body
    return decode_samples_and_merge_input_sample(response.body, Sample()).samples


class CapturingHook:
    """A stand-in for a real capture producer: one artifact per session, named on every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def __call__(self, *, session_id, sequence, input_ids, request_body):
        self.calls.append((sequence, len(input_ids)))
        reference = {"kind": "test-capture", "capture_id": f"cap-{session_id}"}
        request_body["rid"] = f"bs:{reference['capture_id']}:{session_id}:{sequence}"
        return reference


def test_no_hook_leaves_the_request_and_the_sample_untouched():
    backend = RecordingBackend()
    core = build_core(backend)
    session_id = core.registry.create_session()
    chat(core, session_id, [{"role": "user", "content": "hello"}])
    assert "rid" not in backend.bodies[0]
    assert collect(core, session_id)[0].metadata.get(CONDITIONING_METADATA_KEY) is None


def test_the_hook_runs_before_the_proxy_and_names_every_call():
    hook = CapturingHook()
    backend = RecordingBackend()
    with function_registry.temporary("test_conditioning.hook", hook):
        core = build_core(backend, hook_path="test_conditioning.hook")
        session_id = core.registry.create_session()
        chat(core, session_id, [{"role": "user", "content": "hello"}])
        chat(
            core,
            session_id,
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "again"},
            ],
        )
        samples = collect(core, session_id)

    assert [sequence for sequence, _ in hook.calls] == [1, 2], "the sequence is per call attempt"
    assert hook.calls[1][1] > hook.calls[0][1], "the hook sees each call's own input_ids"
    rids = [body["rid"] for body in backend.bodies]
    assert len(set(rids)) == 2, f"every call needs its own rid, got {rids}"
    assert all(rid.startswith(f"bs:cap-{session_id}:") for rid in rids)
    assert samples[0].metadata[CONDITIONING_METADATA_KEY] == {"kind": "test-capture", "capture_id": f"cap-{session_id}"}


def test_an_async_hook_is_awaited_before_the_proxy():
    backend = RecordingBackend()
    order: list[str] = []

    async def hook(*, session_id, sequence, input_ids, request_body):
        await asyncio.sleep(0)
        order.append("hook")
        request_body["rid"] = "bs:async"
        return {"capture_id": "async"}

    class OrderedBackend(RecordingBackend):
        async def do_proxy(self, request, path, *, body, headers):
            order.append("proxy")
            return await super().do_proxy(request, path, body=body, headers=headers)

    backend = OrderedBackend()
    with function_registry.temporary("test_conditioning.async_hook", hook):
        core = build_core(backend, hook_path="test_conditioning.async_hook")
        session_id = core.registry.create_session()
        chat(core, session_id, [{"role": "user", "content": "hello"}])
    assert order == ["hook", "proxy"]
    assert backend.bodies[0]["rid"] == "bs:async"


def test_a_session_that_changes_its_reference_is_an_error():
    references = iter([{"capture_id": "first"}, {"capture_id": "second"}])

    def hook(*, session_id, sequence, input_ids, request_body):
        return next(references)

    backend = RecordingBackend()
    with function_registry.temporary("test_conditioning.drifting", hook):
        core = build_core(backend, hook_path="test_conditioning.drifting")
        session_id = core.registry.create_session()
        chat(core, session_id, [{"role": "user", "content": "hello"}])
        with pytest.raises(ValueError, match="changed its conditioning reference"):
            chat(
                core,
                session_id,
                [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "again"},
                ],
            )
    assert len(backend.bodies) == 1, "a drifted reference must not reach the backend"


def test_a_failing_hook_stops_the_call_before_the_backend():
    def hook(*, session_id, sequence, input_ids, request_body):
        raise RuntimeError("capture store is full")

    backend = RecordingBackend()
    with function_registry.temporary("test_conditioning.failing", hook):
        core = build_core(backend, hook_path="test_conditioning.failing")
        session_id = core.registry.create_session()
        with pytest.raises(RuntimeError, match="capture store is full"):
            chat(core, session_id, [{"role": "user", "content": "hello"}])
    assert backend.bodies == []


def test_apply_conditioning_keeps_the_previous_reference_when_the_hook_returns_none():
    def hook(*, session_id, sequence, input_ids, request_body):
        return None

    previous = {"capture_id": "kept"}
    result = asyncio.run(
        apply_conditioning(hook, session_id="s", sequence=2, input_ids=[1], request_body={}, previous=previous)
    )
    assert result is previous


def test_apply_conditioning_rejects_a_non_object_reference():
    def hook(*, session_id, sequence, input_ids, request_body):
        return "capture-1"

    with pytest.raises(TypeError, match="JSON object"):
        asyncio.run(apply_conditioning(hook, session_id="s", sequence=1, input_ids=[1], request_body={}, previous=None))


def test_stamp_conditioning_preserves_other_metadata():
    sample = Sample()
    sample.metadata = {"reward_source": "grader"}
    stamp_conditioning([sample], {"capture_id": "x"})
    assert sample.metadata == {"reward_source": "grader", CONDITIONING_METADATA_KEY: {"capture_id": "x"}}
    stamp_conditioning([sample], None)
    assert sample.metadata[CONDITIONING_METADATA_KEY] == {"capture_id": "x"}
