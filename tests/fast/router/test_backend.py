from argparse import Namespace
from types import SimpleNamespace

import pytest
from tests.fast.fixtures.args_fixtures import parser_defaults, sglang_router_args

from miles.backends.sglang_utils.sglang_api_client import WorkerType
from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig
from miles.router.backend import _BACKEND_CLASSES, RouterKind, make_router_backend
from miles.utils.args.configs.router import RouterConfig


def _model_cfg(*worker_types: WorkerType) -> ModelConfig:
    groups = [ServerGroupConfig(worker_type=t, num_gpus_per_engine=1, needs_offload=False) for t in worker_types]
    return ModelConfig(name="default", model_path=None, server_groups=groups, update_weights=True)


_REGULAR = _model_cfg(WorkerType.REGULAR)
_PD = _model_cfg(WorkerType.PREFILL, WorkerType.DECODE)


def test_every_router_kind_has_its_own_backend():
    """A kind without a backend would only fail once a run selects it."""
    assert set(_BACKEND_CLASSES) == set(RouterKind)
    assert all(backend.kind is kind for kind, backend in _BACKEND_CLASSES.items())


@pytest.mark.parametrize(
    "router, policy, expected",
    [
        ("sglang", "consistent_hashing", {"X-SMG-Routing-Key": "key-1"}),
        ("sglang", "manual", {"X-SMG-Routing-Key": "key-1"}),
        ("sglang", "cache_aware", {}),
        ("sglang", None, {}),
        ("miles", "manual", {}),
    ],
    ids=[
        "sgl-router-consistent-hashing",
        "sgl-router-manual",
        "sgl-router-cache-aware",
        "sgl-router-default",
        "miles-router",
    ],
)
def test_a_routing_key_header_is_sent_only_to_a_router_that_reads_it(router, policy, expected):
    backend = make_router_backend(Namespace(**sglang_router_args(policy) | {"router": router}))

    assert backend.routing_headers("key-1", [_REGULAR]) == expected


def test_a_key_routed_request_without_a_key_is_rejected():
    """Without the header a key policy falls back to an implicit key and can pin every request to one engine."""
    backend = make_router_backend(Namespace(**sglang_router_args("manual")))

    with pytest.raises(ValueError, match="routes by routing key"):
        backend.routing_headers(None, [_REGULAR])


@pytest.mark.parametrize(
    "model_cfg, policies, expected",
    [
        (_REGULAR, {"router_policy": "manual"}, True),
        (_REGULAR, {"router_policy": "manual", "sglang_router_policy": "cache_aware"}, False),
        (_REGULAR, {"router_prefill_policy": "manual"}, False),
        (_PD, {"router_prefill_policy": "consistent_hashing"}, True),
        (_PD, {"router_decode_policy": "manual"}, True),
        (_PD, {"sglang_router_policy": "manual"}, True),
        (
            _PD,
            {
                "sglang_router_policy": "manual",
                "router_prefill_policy": "cache_aware",
                "router_decode_policy": "cache_aware",
            },
            False,
        ),
    ],
    ids=[
        "pass-through-policy",
        "sglang-router-policy-overrides-pass-through",
        "non-pd-ignores-pool-policy",
        "pd-prefill-policy",
        "pd-decode-policy",
        "pd-pools-fall-back-to-main-policy",
        "pd-pools-override-main-policy",
    ],
)
def test_routing_by_key_follows_the_policies_the_router_runs(model_cfg, policies, expected):
    """Only the policies sgl-router applies decide: the main one outside PD, each pool's own (or the main) in PD."""
    flags = parser_defaults() | policies
    backend = make_router_backend(Namespace(**flags | RouterConfig.from_args(Namespace(**flags))))

    assert backend.routes_by_key([model_cfg]) is expected


@pytest.mark.parametrize(
    "external_pd, policies, expected",
    [
        (False, {"sglang_router_policy": "manual"}, True),
        (False, {"router_prefill_policy": "manual"}, False),
        (True, {"router_prefill_policy": "manual"}, True),
    ],
    ids=["external-router", "external-router-ignores-pool-policy", "external-pd-router"],
)
def test_an_external_router_reads_pd_from_its_flag(external_pd, policies, expected):
    """Without a launched engine there is no model config, so --rollout-external-router-pd says whether PD runs."""
    flags = parser_defaults() | policies | {"rollout_external_router_pd": external_pd}
    args = Namespace(**flags | RouterConfig.from_args(Namespace(**flags)), sglang=SimpleNamespace(models=[]))

    assert make_router_backend(args).routes_by_key(args.sglang.models) is expected


@pytest.mark.parametrize(
    "model_cfgs, expected",
    [([_PD], False), ([_PD, _REGULAR], True), ([_REGULAR, _PD], True)],
    ids=["pd-pools-only", "pd-then-regular", "regular-then-pd"],
)
def test_a_key_is_sent_when_any_models_router_reads_it(model_cfgs, expected):
    """Requests may target any model's router, so one router running a key policy is enough to send the key."""
    flags = parser_defaults() | {
        "sglang_router_policy": "manual",
        "router_prefill_policy": "cache_aware",
        "router_decode_policy": "cache_aware",
    }
    backend = make_router_backend(Namespace(**flags | RouterConfig.from_args(Namespace(**flags))))

    assert backend.routes_by_key(model_cfgs) is expected


def _ricky_group(**overrides: str) -> ServerGroupConfig:
    return ServerGroupConfig(
        worker_type=WorkerType.REGULAR,
        num_gpus_per_engine=1,
        needs_offload=False,
        overrides={"model_path": "/models/ricky", "franky_side": "ricky", **overrides},
    )


_SANDY_GROUP = ServerGroupConfig(
    worker_type=WorkerType.REGULAR,
    num_gpus_per_engine=2,
    needs_offload=False,
    overrides={"model_path": "/models/sandy", "franky_side": "sandy"},
)


def _franky_model_cfg(*server_groups: ServerGroupConfig) -> ModelConfig:
    return ModelConfig(name="franky", model_path=None, server_groups=list(server_groups), update_weights=True)


def _sgl_router_argv(
    model_cfg: ModelConfig, *, sgl_router_path: str = "sgl-router", sgl_router_args: str = ""
) -> list[str]:
    args = Namespace(
        router="sgl_router",
        sgl_router_path=sgl_router_path,
        sgl_router_args=sgl_router_args,
        sglang_router_request_timeout_secs=600,
        rollout_external_router_pd=False,
    )
    ctx = SimpleNamespace(self_addrs={"primary": SimpleNamespace(host="127.0.0.1", port=30000)})
    return make_router_backend(args).launch_argv(ctx, model_cfg=model_cfg, interpreter_prefix=["python"])


def test_sgl_router_serves_the_ricky_model_and_takes_workers_over_its_api():
    argv = _sgl_router_argv(
        _franky_model_cfg(_ricky_group(), _SANDY_GROUP),
        sgl_router_path="/bin/sgl-router",
        sgl_router_args="--franky-block-size 256 --policy cache_aware",
    )

    assert argv == [
        *("/bin/sgl-router", "--host", "127.0.0.1", "--port", "30000"),
        *("--model-id", "/models/ricky", "--tokenizer-path", "/models/ricky/tokenizer.json"),
        *("--worker-api", "--request-timeout-secs", "600"),
        *("--franky-block-size", "256", "--policy", "cache_aware"),
    ]


def test_sgl_router_serves_the_name_the_ricky_engines_are_served_under():
    argv = _sgl_router_argv(_franky_model_cfg(_ricky_group(served_model_name="policy"), _SANDY_GROUP))

    assert argv[argv.index("--model-id") + 1] == "policy"
    assert argv[argv.index("--tokenizer-path") + 1] == "/models/ricky/tokenizer.json"


@pytest.mark.parametrize(
    "server_groups, message",
    [
        (
            [_ricky_group(served_model_name="policy"), _ricky_group(served_model_name="agent"), _SANDY_GROUP],
            "sgl-router needs one served_model_name for model 'franky', but its Ricky server groups have "
            "['agent', 'policy']",
        ),
        (
            [
                _ricky_group(served_model_name="policy"),
                _ricky_group(model_path="/models/other", served_model_name="policy"),
                _SANDY_GROUP,
            ],
            "sgl-router needs one model_path for model 'franky', but its Ricky server groups have "
            "['/models/other', '/models/ricky']",
        ),
        (
            [_SANDY_GROUP],
            "sgl-router needs one served_model_name for model 'franky', but its Ricky server groups have []",
        ),
    ],
    ids=["two-served-names", "two-model-paths", "no-ricky-group"],
)
def test_sgl_router_refuses_ricky_groups_without_one_shared_value(server_groups, message):
    with pytest.raises(ValueError) as exc_info:
        _sgl_router_argv(_franky_model_cfg(*server_groups))

    assert str(exc_info.value) == message


def test_sgl_router_takes_no_name_from_a_placeholder_group():
    placeholder = ServerGroupConfig(
        worker_type=WorkerType.PLACEHOLDER,
        num_gpus_per_engine=1,
        needs_offload=False,
        overrides={"model_path": "/models/ricky"},
    )

    argv = _sgl_router_argv(_franky_model_cfg(_ricky_group(served_model_name="policy"), placeholder, _SANDY_GROUP))

    assert argv[argv.index("--model-id") + 1] == "policy"
