from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from tests.fast.fixtures.session_fixtures import make_session_server_config
from tests.fast.router.test_linear_trajectory import _make_registry

from miles.rollout.session.errors import MessageValidationError, TokenizationError
from miles.rollout.session.types import SessionRecord


USER = {"role": "user", "content": "first"}
ASSISTANT = {"role": "assistant", "content": "answer"}
TOOL = {"role": "tool", "content": "result", "tool_call_id": "call_1"}


def _snapshot(session):
    return deepcopy({key: value for key, value in vars(session).items() if key != "lock"})


def _commit(session, messages, prompt, completion):
    session.update_pretokenized_state(messages, ASSISTANT, prompt, completion, max_trim_tokens=0)
    session.append_record(
        SessionRecord(
            timestamp=1.0,
            method="POST",
            path="/v1/chat/completions",
            status_code=200,
            request={"messages": messages},
            response={},
        )
    )


@pytest.mark.parametrize("checkpoint_count", [1, 2])
@pytest.mark.parametrize("failure", ["input_ids", "kwargs", "lora", "model", "render"])
def test_failed_retry_preparation_preserves_history(checkpoint_count, failure):
    registry = _make_registry()
    session = registry.get_session(registry.create_session())
    _commit(session, [USER], [1], [2])
    messages = [USER]
    if checkpoint_count == 2:
        messages = [USER, ASSISTANT, TOOL]
        _commit(session, messages, [1, 2, 3], [4])
    before = _snapshot(session)
    client_args = {"messages": messages}
    error_type = MessageValidationError
    if failure == "input_ids":
        client_args["input_ids"] = [999]
        error = "input_ids=.*is not accepted"
    elif failure == "kwargs":
        client_args["chat_template_kwargs"] = []
        error = "chat_template_kwargs must be an object"
    elif failure == "lora":
        client_args["lora_path"] = "wrong-adapter"
        error = "lora_path=.*is not accepted"
    elif failure == "model":
        error = "model rejected request"
        registry.tito_tokenizer.resolve_request_args = MagicMock(side_effect=ValueError(error))
    else:
        error = "render failed"
        error_type = TokenizationError
        render = "apply_chat_template" if checkpoint_count == 1 else "merge_tokens"
        setattr(registry.tito_tokenizer, render, MagicMock(side_effect=TokenizationError(error)))

    with pytest.raises(error_type, match=error):
        session.prepare_token_ids_and_request_args(
            client_args, config=make_session_server_config(), tito_tokenizer=registry.tito_tokenizer
        )

    assert _snapshot(session) == before


def test_valid_retry_renders_selected_checkpoint_before_rollback():
    registry = _make_registry()
    session = registry.get_session(registry.create_session())
    _commit(session, [USER], [1], [2])
    checkpoint = _snapshot(session)
    _commit(session, [USER, ASSISTANT, TOOL], [1, 2, 3], [4])
    before = _snapshot(session)
    retry = [USER, ASSISTANT, {**TOOL, "content": "retry"}]

    def render(*, old_messages, new_messages, pretokenized_token_ids, template_args):
        assert _snapshot(session) == before
        assert old_messages == checkpoint["messages"]
        assert pretokenized_token_ids == [1, 2]
        assert new_messages == retry
        return [1, 2, 5]

    registry.tito_tokenizer.merge_tokens = MagicMock(side_effect=render)
    prepared = session.prepare_token_ids_and_request_args(
        {"messages": retry}, config=make_session_server_config(), tito_tokenizer=registry.tito_tokenizer
    )

    assert prepared.body["input_ids"] == [1, 2, 5]
    assert _snapshot(session) == checkpoint


def test_each_turn_names_the_earlier_answers_it_still_holds_for_franky_engines():
    """A Franky engine computes an earlier answer again with the taps its decode read, so it must know where it is."""
    registry = _make_registry()
    session = registry.get_session(registry.create_session())
    config = make_session_server_config(franky_generated_spans=True)
    _commit(session, [USER], [1, 2], [3, 4])
    _commit(session, [USER, ASSISTANT, TOOL], [1, 2, 3, 4, 5], [6, 7, 8])
    registry.tito_tokenizer.merge_tokens = MagicMock(return_value=[1, 2, 3, 4, 5, 6, 7, 9])
    turn = [USER, ASSISTANT, TOOL, ASSISTANT, {**TOOL, "content": "next"}]

    prepared = session.prepare_token_ids_and_request_args(
        {"messages": turn}, config=config, tito_tokenizer=registry.tito_tokenizer
    )

    assert prepared.body["franky_generated_spans"] == [[2, 4], [5, 7]]


def test_a_turn_without_franky_engines_sends_no_answers():
    registry = _make_registry()
    session = registry.get_session(registry.create_session())
    _commit(session, [USER], [1, 2], [3, 4])
    registry.tito_tokenizer.merge_tokens = MagicMock(return_value=[1, 2, 3, 4, 5])

    prepared = session.prepare_token_ids_and_request_args(
        {"messages": [USER, ASSISTANT, TOOL]}, config=make_session_server_config(), tito_tokenizer=registry.tito_tokenizer
    )

    assert "franky_generated_spans" not in prepared.body
