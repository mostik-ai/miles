from argparse import Namespace

import pytest
import torch

from miles.backends.training_utils.data import context_parallel
from miles.backends.training_utils.data import rollout as data_utils
from miles.backends.training_utils.parallel import GroupInfo, ParallelState


def _parallel_state() -> ParallelState:
    trivial_group = GroupInfo(rank=0, size=1, group=None)
    return ParallelState(
        intra_dp=trivial_group,
        intra_dp_cp=trivial_group,
        cp=trivial_group,
        tp=trivial_group,
        pp=trivial_group,
        ep=trivial_group,
        etp=trivial_group,
        indep_dp=trivial_group,
    )


def _args(*, micro_batch_size: int, use_dynamic_batch_size: bool) -> Namespace:
    return Namespace(
        enable_witness=False,
        qkv_format="bshd",
        data_pad_size_multiplier=1,
        compress_ratios=[4],
        micro_batch_size=micro_batch_size,
        use_dynamic_batch_size=use_dynamic_batch_size,
        true_on_policy_mode=False,
        bf16=False,
        fp16=False,
    )


def _rollout(total_lengths: list[int]) -> dict:
    return {
        "tokens": [list(range(length)) for length in total_lengths],
        "loss_masks": [[1]] * len(total_lengths),
        "total_lengths": total_lengths,
        "response_lengths": [1] * len(total_lengths),
    }


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
    monkeypatch: pytest.MonkeyPatch,
    micro_batch_size: int,
    use_dynamic_batch_size: bool,
    schedule_columns: dict,
    expected: list[int],
) -> None:
    parallel_state = _parallel_state()
    rollout_data = _rollout([5, 11, 3, 7]) | schedule_columns
    monkeypatch.setattr(data_utils, "process_rollout_data", lambda *args, **kwargs: (rollout_data, object()))
    monkeypatch.setattr(data_utils, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(context_parallel, "get_parallel_state", lambda: parallel_state)
    monkeypatch.setattr(data_utils.current_platform, "current_device", lambda: torch.device("cpu"))

    batch, _ = data_utils.get_rollout_data(
        args=_args(micro_batch_size=micro_batch_size, use_dynamic_batch_size=use_dynamic_batch_size),
        rollout_data_ref=object(),
        train_parallel_config=parallel_state.train_parallel_config(supports_precomputed_schedule=True),
    )

    assert batch["max_seq_lens"] == expected
