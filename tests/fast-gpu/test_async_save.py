"""Megatron async saves after SGLang has patched torch's tensor reducers."""

from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(
    est_time=120, suite="stage-b-2-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"], num_gpus=1
)

import os
import subprocess
import sys
from pathlib import Path

import pytest

_WORKER = Path(__file__).with_name("_async_save_worker.py")
_REPO_ROOT = Path(__file__).parents[2]


def test_async_save_after_sglang_patches_torch_reducers_completes_and_loads_back(tmp_path: Path) -> None:
    pytest.importorskip("nvidia_resiliency_ext")
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "PYTHONPATH": os.pathsep.join(filter(None, [str(_REPO_ROOT), os.environ.get("PYTHONPATH")])),
    }
    log = tmp_path / "worker.log"
    # a file, not a pipe: the checkpoint worker inherits it; a save that never reaches that worker blocks forever
    with log.open("w") as output:
        try:
            code = subprocess.run(
                [sys.executable, str(_WORKER), "--directory", str(tmp_path)],
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                timeout=300,
            ).returncode
        except subprocess.TimeoutExpired:
            code = "timeout"
    assert code == 0 and "PASS" in log.read_text(), f"exit {code}:\n{log.read_text()[-5000:]}"
