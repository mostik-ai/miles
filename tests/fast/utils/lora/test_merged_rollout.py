from argparse import Namespace

import pytest

from miles.utils.lora.arguments import validate_lora_args
from miles.utils.lora.utils import lora_rollout_enabled


def merged_args(**overrides) -> Namespace:
    fields = {
        "lora_rank": 8,
        "lora_adapter_path": None,
        "train_backend": "megatron",
        "megatron_to_hf_mode": "bridge",
        "lora_checkpoint_format": "adapter",
        "multi_lora_n_adapters": 0,
        "lora_train_only": False,
        "lora_merged_rollout": True,
        "use_session_server": False,
    }
    return Namespace(**(fields | overrides))


def test_a_merged_rollout_serves_no_adapter_so_engines_and_weight_sync_take_the_full_weight_path():
    assert not lora_rollout_enabled(merged_args())
    assert lora_rollout_enabled(merged_args(lora_merged_rollout=False))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"megatron_to_hf_mode": "raw"}, "Bridge's export"),
        ({"lora_train_only": True}, "frozen base"),
        ({"use_session_server": True}, "session server"),
    ],
)
def test_a_merged_rollout_refuses_modes_that_would_serve_something_else(overrides, message):
    with pytest.raises(AssertionError, match=message):
        validate_lora_args(merged_args(**overrides))
