import argparse
from collections.abc import Mapping

from sglang_router.launch_router import RouterArgs

from miles.utils.cli_dest_utils import dests_on_command_line
from miles.utils.workers.argv_utils import render_cli_argv

_ROUTER_DEST_PREFIX = "router_"


def compute_sglang_router_args(
    args: argparse.Namespace,
    *,
    host: str,
    port: int,
    prometheus_port: int,
    has_pd_disaggregation: bool,
) -> dict[str, object]:
    """The router's launch arguments, with the allocator owning its address and nothing else.

    Raises:
        AssertionError: a setting two flags spell is spelled differently by both of them.
    """
    named = dests_on_command_line()
    router_args = _extract_router_cli_values(args)
    router_args.update(
        host=host,
        port=port,
        prometheus_port=prometheus_port,
        request_timeout_secs=_single_owner(
            args, named, legacy="sglang_router_request_timeout_secs", prefixed="router_request_timeout_secs"
        ),
    )
    if "router_log_level" not in named:
        # The engines already log every request; the router repeating them drowns the rollout log.
        router_args["log_level"] = "warn"

    if policy := _single_owner(args, named, legacy="sglang_router_policy", prefixed="router_policy"):
        router_args["policy"] = policy

    if has_pd_disaggregation:
        router_args["pd_disaggregation"] = True

    return router_args


def _single_owner(args: argparse.Namespace, named: frozenset[str], *, legacy: str, prefixed: str) -> object:
    """The value of a router setting that both a legacy ``--sglang-router-*`` flag and ``--router-*`` spell."""
    legacy_value, prefixed_value = getattr(args, legacy), getattr(args, prefixed, None)
    assert not (legacy in named and prefixed in named and legacy_value != prefixed_value), (
        f"--{legacy.replace('_', '-')} and --{prefixed.replace('_', '-')} name the same router setting as "
        f"{legacy_value!r} and {prefixed_value!r}; only one of the two can own it"
    )
    return prefixed_value if prefixed in named and legacy not in named else legacy_value


def router_args_to_argv(router_args: Mapping[str, object]) -> list[str]:
    return render_cli_argv(
        router_args,
        expected_obj=RouterArgs.from_cli_args(argparse.Namespace(**router_args)),
        make_parser=_make_cli_parser,
        from_parsed=RouterArgs.from_cli_args,
    )


def parse_router_args_argv(argv: list[str]) -> RouterArgs:
    return RouterArgs.from_cli_args(_make_cli_parser().parse_args(argv))


def _make_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser)
    return parser


def _extract_router_cli_values(args: argparse.Namespace) -> dict[str, object]:
    prefixed_defaults = vars(_make_prefixed_cli_parser().parse_args([]))
    return {name.removeprefix(_ROUTER_DEST_PREFIX): getattr(args, name) for name in prefixed_defaults}


def _make_prefixed_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser, use_router_prefix=True, exclude_host_port=True)
    return parser
