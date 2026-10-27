import ast
import copy
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def actor_type():
    root = Path(__file__).resolve().parents[4]
    actor_path = root / "miles/backends/megatron_utils/actor.py"
    tree = ast.parse(actor_path.read_text())
    actor = copy.deepcopy(
        next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor")
    )
    selected = {"_weight_sync_reads_tms_backup", "_enable_weight_backup", "_get_actor_weights"}
    actor.body = [node for node in actor.body if isinstance(node, ast.FunctionDef) and node.name in selected]
    assert {node.name for node in actor.body} == selected
    actor.bases = []
    actor.keywords = []
    actor.decorator_list = []
    lora_path = root / "miles/utils/lora/utils.py"
    lora_tree = ast.parse(lora_path.read_text())
    enabled = next(
        node for node in lora_tree.body if isinstance(node, ast.FunctionDef) and node.name == "is_lora_enabled"
    )
    namespace = {"Namespace": Namespace}
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[enabled], type_ignores=[])), str(lora_path), "exec"),
        namespace,
    )
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=[actor], type_ignores=[])), str(actor_path), "exec"),
        namespace,
    )
    return namespace["MegatronTrainRayActor"]


@pytest.mark.parametrize(
    ("colocate", "offload", "rank", "target", "active", "reference", "asleep", "route", "backup"),
    [
        (False, False, 32, "cpu", "actor", False, False, "live", False),
        (True, False, 32, "cpu", "actor", False, False, "live", False),
        (True, False, 0, "cpu", "actor", False, False, "live", False),
        (True, True, 32, "cpu", "actor", False, False, "memory_saver", False),
        (True, True, 32, "cpu", "actor", False, True, "memory_saver", False),
        (True, True, 0, "cpu", "actor", False, False, "host", True),
        (True, True, 32, "disk", "actor", False, False, "host", True),
        (True, False, 32, "cpu", "ref", True, False, "host", True),
        (True, False, 32, "cpu", "actor", True, False, "live", True),
        (True, True, 32, "cpu", "actor", True, False, "host", True),
        (False, False, 32, "cpu", "ref", True, False, "host", True),
        (False, True, 32, "cpu", "actor", False, True, "host", True),
        (False, True, 32, "cpu", "actor", False, False, "live", True),
    ],
)
def test_weight_sync_reads_the_owned_actor_storage(
    actor_type, colocate, offload, rank, target, active, reference, asleep, route, backup
):
    worker = actor_type()
    worker.args = Namespace(
        colocate=colocate,
        offload_train=offload,
        offload_train_target=target,
        lora_rank=rank,
        keep_old_actor=False,
    )
    worker.with_ref = reference
    worker.with_opd_teacher = False
    worker._asleep = asleep
    worker._active_model_tag = active
    weights = {name: object() for name in ("live", "host", "memory_saver")}
    reads = []

    def named_weights(*, translate_gpu_to_cpu=False):
        if translate_gpu_to_cpu and not offload:
            raise AssertionError("TorchMemorySaver observes invalid LD_PRELOAD")
        selected = "memory_saver" if translate_gpu_to_cpu else "live"
        reads.append(selected)
        return {"weight": weights[selected]}.items()

    def host_weights(tag):
        assert tag == "actor"
        reads.append("host")
        return {"weight": weights["host"]}

    worker._named_actor_weights = named_weights
    worker.weights_backuper = SimpleNamespace(get=host_weights)

    result = worker._get_actor_weights()

    assert result["weight"] is weights[route]
    assert reads == [route]
    assert worker._enable_weight_backup is backup
