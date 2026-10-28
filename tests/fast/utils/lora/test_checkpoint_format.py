from argparse import Namespace

import pytest

from miles.utils.lora.arguments import validate_lora_args


def lora_args(megatron_to_hf_mode: str, multi_lora_n_adapters: int = 0) -> Namespace:
    return Namespace(
        lora_rank=8,
        lora_adapter_path=None,
        train_backend="megatron",
        megatron_to_hf_mode=megatron_to_hf_mode,
        lora_checkpoint_format="torch_dist",
        multi_lora_n_adapters=multi_lora_n_adapters,
        colocate=False,
        offload_train=False,
        ft_components=[],
        calculate_per_token_loss=False,
    )


def test_torch_dist_lora_checkpoints_refuse_native_adapters_that_megatron_does_not_save():
    with pytest.raises(AssertionError, match="--megatron-to-hf-mode bridge"):
        validate_lora_args(lora_args("raw"))


def test_torch_dist_lora_checkpoints_refuse_multi_lora_slots():
    with pytest.raises(AssertionError, match="one adapter"):
        validate_lora_args(lora_args("bridge", multi_lora_n_adapters=2))
