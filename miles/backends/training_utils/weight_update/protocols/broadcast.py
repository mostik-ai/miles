import socket
from argparse import Namespace
from collections.abc import Sequence
from concurrent.futures import Future
from contextlib import AbstractContextManager, ExitStack, nullcontext

import ray
import torch
import torch.distributed as dist

from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient
from miles.backends.training_utils.parallel import ParallelState, get_parallel_state
from miles.backends.training_utils.weight_update.hf_weight_iterator import WeightUpdatePlacement
from miles.backends.training_utils.weight_update.protocol import WeightTransferProtocol
from miles.backends.training_utils.weight_update.utils import get_data_replica_rank_and_size
from miles.utils import async_utils
from miles.utils.distributed_lock import create_world_ticket_lock
from miles.utils.distributed_utils import init_process_group


class UpdateWeightFromDistributed(WeightTransferProtocol):
    """
    Update distributed engines via NCCL. Each PP rank: group "miles-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
    """

    supports_lora = True

    def __init__(self, args: Namespace) -> None:
        super().__init__(args)
        self._model_update_groups = None
        # --update-weight-pipelined: the one posted bucket, with what must outlive it (its RPC futures, the tensors
        # its broadcasts read, their handles and the held engine lock) until both engines acked it.
        self._in_flight: tuple | None = None
        # --update-weight-flattened: two byte slots used in turn, so the slot a bucket is packed into was last read
        # by the bucket before the previous one, which has been acked.
        self._slots: list[torch.Tensor | None] = [None, None]
        self._next_slot = 0
        parallel_state = get_parallel_state()
        self._engine_lock: AbstractContextManager = (
            create_world_ticket_lock(
                prefix="miles/weight_update",
                participates=parallel_state.intra_dp_cp.rank == 0 and parallel_state.tp.rank == 0,
            )
            if parallel_state.pp.size > 1
            else nullcontext()
        )

    def connect(
        self,
        rollout_engines: Sequence[SGLangApiClient],
        engine_gpu_counts: Sequence[int] | None,
        engine_gpu_offsets: Sequence[int] | None,
        parallel_state: ParallelState,
        placement: WeightUpdatePlacement,
        selector: str,
    ) -> None:
        """
        Create NCCL "miles-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self._selector = selector
        self._engine_gpu_counts = engine_gpu_counts

        # One sender per replica set; one NCCL group (sender + all engines) per shard.
        replica_rank, _ = get_data_replica_rank_and_size(parallel_state, placement)
        self.is_sender = replica_rank == 0
        shard = 0 if placement.gather_pp else parallel_state.pp.rank
        if self.is_sender:
            self.group_name = f"miles-pp_{shard}"
            disconnect_rollout_engines_from_distributed(
                self.args, self.group_name, self._model_update_groups, self.rollout_engines
            )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, self.group_name, rollout_engines, engine_gpu_counts=engine_gpu_counts
            )

    def send_bucket(self, bucket: list[tuple[str, torch.Tensor]]) -> None:
        """Lock → broadcast → clear → unlock. Lock prevents NCCL deadlock.

        Pipelined, the bucket is posted and the call returns once the previous one is acked and released, so the
        caller exports the next bucket while this one moves; the lock is held until this one is acked. Flattened, the
        bucket is packed into a slot, even before the previous bucket's acks (its slot is the other one), and its own
        tensors are dropped once posted; the slot is what stays in flight.
        """
        flat = self._pack(bucket) if self.args.update_weight_flattened else None
        if self.args.update_weight_pipelined:
            self._release_in_flight()
            with ExitStack() as held:
                held.enter_context(self._engine_lock)
                posted = post_weights_from_distributed(
                    self.group_name, self._model_update_groups, self.rollout_engines, bucket, self._selector, flat
                )
                if flat is not None:
                    bucket.clear()
                self._in_flight = (bucket, *posted, held.pop_all())  # the lock stays held until the acks
            return
        with self._engine_lock:
            futures = update_weights_from_distributed(
                self.group_name,
                self._model_update_groups,
                self.rollout_engines,
                bucket,
                selector=self._selector,
                flat=flat,
            )
            async_utils.wait_futures(futures)
            bucket.clear()

    def _pack(self, bucket: list[tuple[str, torch.Tensor]]) -> torch.Tensor:
        """Reorder the bucket as it will be packed and pack it into the next slot. A slot is allocated at the bucket
        size limit, so it is reused all sync; only a single unit over the limit grows it."""
        bucket[:] = flattened_order(bucket)
        size = sum(tensor.numel() * tensor.element_size() for _, tensor in bucket)
        index, self._next_slot = self._next_slot, 1 - self._next_slot
        if self._slots[index] is None or self._slots[index].numel() < size:
            self._slots[index] = None  # freed before its replacement is allocated
            self._slots[index] = torch.empty(
                max(size, self.args.update_weight_buffer_size), dtype=torch.uint8, device=bucket[0][1].device
            )
        return pack_flattened(bucket, self._slots[index])

    def begin_sync(self, weight_version, iter_buckets) -> bool:
        """A bucket a failed sync left in flight is dropped and its lock released, so a retry starts clean instead
        of re-raising that failure or waiting on the lock."""
        if self._in_flight is not None:
            *_, lock = self._in_flight
            self._in_flight = None
            lock.close()
        self._release_slots()  # a dropped bucket's broadcast may still read its slot
        return True

    def after_base_weights(self) -> None:
        """The last posted bucket is acked and released before the session is finalized; the slots go back to
        training until the next sync."""
        self._release_in_flight()
        self._release_slots()

    def _release_slots(self) -> None:
        self._slots, self._next_slot = [None, None], 0

    def _release_in_flight(self) -> None:
        """Wait for the posted bucket's broadcasts and both engines' acks, then drop it and the lock; an engine's
        failure raises here, before anything else is posted."""
        if self._in_flight is None:
            return
        bucket, futures, _tensors, handles, lock = self._in_flight
        self._in_flight = None
        with lock:
            for handle in handles:
                handle.wait()
            async_utils.wait_futures(futures)
        bucket.clear()


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[SGLangApiClient],
    engine_gpu_counts: Sequence[int] | None = None,
) -> dist.ProcessGroup:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)
    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1

    futures = []
    rank_cursor = 1
    for i, api_client in enumerate(rollout_engines):
        futures.append(
            async_utils.submit(
                api_client.init_weights_update_group(
                    master_address,
                    master_port,
                    rank_cursor,
                    world_size,
                    group_name,
                    backend="nccl",
                )
            )
        )
        rank_cursor += engine_gpu_counts[i]
    model_update_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    async_utils.wait_futures(futures)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
    """
    futures = [async_utils.submit(client.destroy_weights_update_group(group_name)) for client in rollout_engines]
    try:
        if model_update_groups is not None:
            dist.destroy_process_group(model_update_groups)
    finally:
        async_utils.wait_futures(futures)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    rollout_engines: Sequence[SGLangApiClient],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    selector: str = "all",
    flat: torch.Tensor | None = None,
) -> list[Future]:
    """
    Send metadata (HTTP), broadcast tensors (NCCL rank 0 → engines).
    """
    futures, _tensors, handles = post_weights_from_distributed(
        group_name, group, rollout_engines, converted_named_tensors, selector=selector, flat=flat
    )
    for handle in handles:
        handle.wait()
    return futures


def post_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    rollout_engines: Sequence[SGLangApiClient],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    selector: str = "all",
    flat: torch.Tensor | None = None,
) -> tuple[list[Future], list[torch.Tensor], list]:
    """Send metadata (HTTP) and launch the broadcasts, waiting for neither: the RPC futures, the contiguous tensors
    the broadcasts read (they must outlive them) and the broadcast handles.

    With ``flat``, the tensors packed by ``pack_flattened`` in this order, one broadcast carries the whole bucket and
    the engines receive it as a flattened bucket.
    """
    packed = {} if flat is None else {"load_format": "flattened_bucket"}
    futures = [
        async_utils.submit(
            client.update_weights_from_distributed(
                names=[name for name, _ in converted_named_tensors],
                dtypes=[param.dtype for _, param in converted_named_tensors],
                shapes=[param.shape for _, param in converted_named_tensors],
                selector=selector,
                group_name=group_name,
                **packed,
            )
        )
        for client in rollout_engines
    ]

    if flat is not None:
        contiguous_tensors = [flat]
    else:
        contiguous_tensors = [
            param.data if param.data.is_contiguous() else param.data.contiguous() for _, param in converted_named_tensors
        ]
    handles = []
    for tensor in contiguous_tensors:
        handles.append(dist.broadcast(tensor, 0, group=group, async_op=True))
    return futures, contiguous_tensors, handles


def flattened_order(bucket: Sequence[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
    """Widest element first, in order otherwise: packed back to back, every tensor then starts at a multiple of its
    element size, which viewing its bytes as its dtype requires."""
    return sorted(bucket, key=lambda item: -item[1].element_size())


def pack_flattened(bucket: Sequence[tuple[str, torch.Tensor]], slot: torch.Tensor) -> torch.Tensor:
    """Copy the bucket's tensors back to back into the byte slot, exactly as SGLang's FlattenedTensorBucket lays
    them out (no padding), and return the used prefix; unaligned tensors are refused."""
    offset = 0
    for name, tensor in bucket:
        width = tensor.element_size()
        if offset % width:
            raise ValueError(f"{name} would start at byte {offset}, not a multiple of its element size {width}")
        size = tensor.numel() * width
        slot[offset : offset + size].copy_(tensor.reshape(-1).view(torch.uint8))
        offset += size
    return slot[:offset]
