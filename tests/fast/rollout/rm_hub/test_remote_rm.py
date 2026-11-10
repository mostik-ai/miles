import asyncio
import json
import threading
import time
from argparse import Namespace
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest

from miles.rollout.rm_hub import remote_rm
from miles.utils.types import Sample

REPLY = {"reward": 1.0}
SHORT_RM_TIMEOUT_SECONDS = 0.2
LONG_RM_TIMEOUT_SECONDS = 60.0
SLOW_REPLY_SECONDS = 1.0
DEFAULT_CONNECT_LIMIT_SECONDS = 0.2
BLOCKED_LOOP_SECONDS = 1.0


def _reward_handler(reply_delay_seconds: float) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(reply_delay_seconds)
            body = json.dumps(REPLY).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    return _Handler


def _serve(reply_delay_seconds: float) -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _reward_handler(reply_delay_seconds))
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def rm_url() -> Iterator[str]:
    yield from _serve(reply_delay_seconds=0.0)


@pytest.fixture
def slow_rm_url() -> Iterator[str]:
    yield from _serve(reply_delay_seconds=SLOW_REPLY_SECONDS)


def _sample() -> Sample:
    return Sample(prompt="", response="answer", label="answer")


async def _reward_while_the_loop_is_blocked(args: Namespace) -> dict:
    call = asyncio.ensure_future(remote_rm(args, _sample()))
    asyncio.get_running_loop().call_soon(time.sleep, BLOCKED_LOOP_SECONDS)
    return await call


def test_rm_timeout_limits_the_whole_call(slow_rm_url):
    args = Namespace(rm_url=slow_rm_url, rm_timeout=SHORT_RM_TIMEOUT_SECONDS)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(remote_rm(args, _sample()))


def test_rm_timeout_lifts_aiohttps_default_connect_limit(rm_url, monkeypatch):
    default = aiohttp.ClientTimeout(total=300, sock_connect=DEFAULT_CONNECT_LIMIT_SECONDS)
    monkeypatch.setattr(aiohttp.client, "DEFAULT_TIMEOUT", default)
    args = Namespace(rm_url=rm_url, rm_timeout=LONG_RM_TIMEOUT_SECONDS)

    assert asyncio.run(_reward_while_the_loop_is_blocked(args)) == REPLY
