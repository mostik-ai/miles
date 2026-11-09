from argparse import Namespace

import pytest

from miles.utils.lora.arguments import validate_lora_args
from miles.utils.lora.utils import lora_rollout_enabled


def lora_args(**overrides) -> Namespace:
    fields = {
        "lora_rank": 8,
        "lora_adapter_path": None,
        "train_backend": "megatron",
        "megatron_to_hf_mode": "bridge",
        "lora_checkpoint_format": "torch_dist",
        "ckpt_format": "torch_dist",
        "multi_lora_n_adapters": 0,
        "colocate": False,
        "offload_train": False,
        "ft_components": [],
        "calculate_per_token_loss": False,
        "lora_train_only": False,
        "lora_merged_rollout": False,
        "use_session_server": False,
    }
    return Namespace(**(fields | overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"megatron_to_hf_mode": "raw"}, "--megatron-to-hf-mode bridge"),
        ({"multi_lora_n_adapters": 2}, "one adapter"),
        ({"ckpt_format": "torch"}, "--ckpt-format torch_dist"),
    ],
)
def test_a_torch_dist_lora_checkpoint_refuses_what_megatron_would_not_save_whole(overrides, message):
    with pytest.raises(AssertionError, match=message):
        validate_lora_args(lora_args(**overrides))


def test_a_merged_rollout_serves_no_adapter_so_engines_and_weight_sync_take_the_full_weight_path():
    assert not lora_rollout_enabled(lora_args(lora_merged_rollout=True))
    assert lora_rollout_enabled(lora_args())


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"megatron_to_hf_mode": "raw"}, "Bridge's export"),
        ({"lora_train_only": True}, "frozen base"),
        ({"use_session_server": True}, "session server"),
        ({"colocate": True}, "colocated trainer"),
    ],
)
def test_a_merged_rollout_refuses_modes_that_would_serve_something_else(overrides, message):
    with pytest.raises(AssertionError, match=message):
        validate_lora_args(lora_args(lora_checkpoint_format="adapter", lora_merged_rollout=True, **overrides))
