from argparse import Namespace
from collections.abc import Callable

import pytest
from tests.fast.backends.training_utils.backend_owner_fixtures import native_backend_owner_state, serve_cpu_rollout

from miles.backends.training_utils.data.rollout import get_data_iterator, get_num_rollouts, get_rollout_data
from miles.backends.training_utils.parallel import GroupInfo, get_parallel_state

__all__ = ["native_backend_owner_state", "serve_cpu_rollout"]


def _rollout(total_lengths: list[int]) -> dict:
    return {
        "partition": list(range(len(total_lengths))),
        "tokens": [list(range(length)) for length in total_lengths],
        "total_lengths": total_lengths,
        "response_lengths": [1] * len(total_lengths),
        "loss_masks": [[1]] * len(total_lengths),
    }


def _bshd_args(
    *, compress_ratios: list[int], micro_batch_size: int = 1, use_dynamic_batch_size: bool = False
) -> Namespace:
    return Namespace(
        backend=Namespace(compress_ratios=compress_ratios, micro_batch_size=micro_batch_size),
        qkv_format="bshd",
        data_pad_size_multiplier=1,
        delay_split_train_data_by_dp=False,
        enable_witness=False,
        multimodal_token_space_processing=False,
        use_dynamic_batch_size=use_dynamic_batch_size,
    )


class TestBackendRolloutSizes:
    @pytest.mark.parametrize("cp_size,ratios,expected", [(1, [4], 12), (2, [4], 16), (2, [], 11)])
    def test_rollout_padding_uses_backend_compression_ratios_and_context_topology(
        self,
        native_backend_owner_state: None,
        serve_cpu_rollout: Callable[[dict], None],
        cp_size: int,
        ratios: list[int],
        expected: int,
    ) -> None:
        """Compression-aware padding grows for context-parallel mirrored regions without flat backend fields."""
        serve_cpu_rollout(_rollout([11]))
        state = get_parallel_state()
        state.cp = GroupInfo(rank=0, size=cp_size, group=None)
        state.intra_dp_cp = GroupInfo(rank=0, size=cp_size, group=None)

        batch, _ = get_rollout_data(
            args=_bshd_args(compress_ratios=ratios),
            rollout_data_ref=[object()],
            train_parallel_config=state.train_parallel_config(supports_precomputed_schedule=True),
        )

        assert batch["max_seq_lens"] == [expected]
        assert batch["total_lengths"] == [11]
        assert batch["tokens"][0].tolist() == list(range(11))

    @pytest.mark.parametrize(
        "micro_batch_size,use_dynamic_batch_size,schedule_columns,expected",
        [
            (1, False, {}, [8, 12, 4, 8]),
            (2, False, {}, [12, 12, 8, 8]),
            (2, False, {"micro_batch_indices": [[0, 2], [1, 3]]}, [8, 12, 8, 12]),
            (1, True, {"micro_batch_indices": [[0, 2], [1, 3]]}, [8, 12, 8, 12]),
            (1, True, {}, [12, 12, 12, 12]),
        ],
    )
    def test_bshd_pads_each_micro_batch_only_to_its_own_longest_row(
        self,
        native_backend_owner_state: None,
        serve_cpu_rollout: Callable[[dict], None],
        micro_batch_size: int,
        use_dynamic_batch_size: bool,
        schedule_columns: dict,
        expected: list[int],
    ) -> None:
        """A short row is not padded to a longer row in another micro-batch; a rollout-side schedule decides the
        micro-batches when it exists, and a training-side dynamic schedule still pads to the rollout's longest."""
        serve_cpu_rollout(_rollout([5, 11, 3, 7]) | schedule_columns)

        batch, _ = get_rollout_data(
            args=_bshd_args(
                compress_ratios=[4], micro_batch_size=micro_batch_size, use_dynamic_batch_size=use_dynamic_batch_size
            ),
            rollout_data_ref=[object()],
            train_parallel_config=get_parallel_state().train_parallel_config(supports_precomputed_schedule=True),
        )

        assert batch["max_seq_lens"] == expected

    @pytest.mark.parametrize("dynamic,steps,rollouts", [(None, [2, 2], [4, 4]), (8, [4], [8])])
    def test_iterator_and_rollout_counts_share_backend_static_and_dynamic_batch_sizes(
        self, native_backend_owner_state: None, dynamic: int | None, steps: list[int], rollouts: list[int]
    ) -> None:
        """The iterator schedule and rollout normalization agree on backend and dynamic batch sizes."""
        args = Namespace(
            backend=Namespace(global_batch_size=4, micro_batch_size=2),
            qkv_format="thd",
            use_dynamic_global_batch_size=dynamic is not None,
            use_dynamic_batch_size=False,
            multimodal_token_space_processing=False,
        )
        data = {"total_lengths": list(range(8))}
        if dynamic is not None:
            data["dynamic_global_batch_size"] = dynamic

        [iterator], counts = get_data_iterator(args=args, model=None, rollout_data=data)

        assert counts == steps
        assert get_num_rollouts(args=args, rollout_data=data, num_steps=len(counts)) == rollouts
        assert [iterator.get_next(["total_lengths"])["total_lengths"] for _ in range(sum(counts))] == [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ]

    def test_multi_lora_rejects_backend_micro_batches_that_cannot_partition_the_local_batch(
        self, native_backend_owner_state: None
    ) -> None:
        """Multi-LoRA cannot silently drop rows from an indivisible fixed micro-batch schedule."""
        args = Namespace(
            backend=Namespace(global_batch_size=4, micro_batch_size=3),
            qkv_format="thd",
            multimodal_token_space_processing=False,
            use_dynamic_global_batch_size=False,
            use_dynamic_batch_size=False,
        )

        with pytest.raises(ValueError, match="local_batch_size=4, micro_batch_size=3"):
            get_data_iterator(args=args, model=None, rollout_data={"total_lengths": [4] * 4, "adapter_slots": [0] * 4})
