"""The session-scoped policy URL an agent function hands to its agent."""

import os


def openai_session_url(base_url: str) -> str:
    """The OpenAI-compatible API root of a session: its base URL plus the ``/v1`` suffix.

    Raises:
        AssertionError: ``MILES_ROUTER_EXTERNAL_HOST`` is set, which used to rewrite the host of every
            session URL out of the ambient environment, where no run record names it.
    """
    assert "MILES_ROUTER_EXTERNAL_HOST" not in os.environ, (
        "MILES_ROUTER_EXTERNAL_HOST rewrote the host of every session URL from the environment, where no "
        "run record names it; place the agent where it can reach the address the allocator handed out"
    )
    return f"{base_url}/v1"
