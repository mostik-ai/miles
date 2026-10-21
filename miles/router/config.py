from miles.utils.pydantic_utils import FrozenStrictBaseModel


class MilesRouterConfig(FrozenStrictBaseModel):
    host: str
    port: int
    max_connections: int
    timeout: float | None
    health_check_interval: float
    health_check_failure_threshold: int
    keep_alive: int = 5
    health_check_timeout: float = 5.0
    # A paused worker's probe timeouts count as failures again after this long without a resume.
    pause_deadline_seconds: float = 1800.0


def compute_miles_router_config(args, *, host: str, port: int, num_engines: int) -> MilesRouterConfig:
    if args.miles_router_max_connections is not None:
        max_connections = args.miles_router_max_connections
    else:
        max_connections = args.sglang_server_concurrency * num_engines

    return MilesRouterConfig(
        host=host,
        port=port,
        max_connections=max_connections,
        timeout=args.miles_router_timeout,
        health_check_interval=args.rollout_health_check_interval,
        health_check_failure_threshold=args.miles_router_health_check_failure_threshold,
        keep_alive=args.miles_router_keep_alive,
        health_check_timeout=args.miles_router_health_check_timeout,
        pause_deadline_seconds=args.miles_router_pause_deadline_seconds,
    )
