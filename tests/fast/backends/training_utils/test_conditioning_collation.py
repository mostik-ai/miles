"""Per-document model inputs keep their document's order through conversion and DP sharding.

A conditioning reference rides in `Sample.multimodal_train_inputs` as one row per document, and
the trainer resolves row `i` as document `i`. That only holds if conversion and DP sharding move
the per-sample entry with its sample; a shift would leave every lookup succeeding while every
document trained on another document's conditioning.
"""

from argparse import Namespace

import torch

from miles.ray.rollout.train_data_conversion import split_train_data_by_dp_raw
from miles.utils.types import Sample

ARGS = Namespace(balance_data=False, multi_lora_n_adapters=0)

KEY = "coral_bigsender_capture"


def _sample(index: int) -> Sample:
    sample = Sample()
    sample.tokens = [1, 2, 3, index]
    sample.response_length = 1
    sample.loss_mask = [1]
    sample.rollout_log_probs = [0.0]
    sample.multimodal_train_inputs = {KEY: torch.tensor([[index, index + 100, index + 200, index + 300]], dtype=torch.int64)}
    return sample


def _rows(entry) -> list[int]:
    return entry[KEY][0].tolist()


def test_dp_sharding_keeps_each_document_with_its_own_model_inputs():
    samples = [_sample(index) for index in range(8)]
    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "loss_masks": [sample.loss_mask for sample in samples],
        "rollout_log_probs": [sample.rollout_log_probs for sample in samples],
        "multimodal_train_inputs": [sample.multimodal_train_inputs for sample in samples],
    }
    shards = split_train_data_by_dp_raw(ARGS, train_data, dp_size=2)
    assert len(shards) == 2
    seen = []
    for shard in shards:
        assert len(shard["multimodal_train_inputs"]) == len(shard["tokens"])
        for tokens, entry in zip(shard["tokens"], shard["multimodal_train_inputs"], strict=True):
            index = tokens[-1]
            assert _rows(entry) == [index, index + 100, index + 200, index + 300]
            seen.append(index)
    assert sorted(seen) == list(range(8)), "every document must survive sharding exactly once"


def test_a_shard_carries_one_entry_per_document_in_document_order():
    samples = [_sample(index) for index in range(4)]
    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        "loss_masks": [sample.loss_mask for sample in samples],
        "rollout_log_probs": [sample.rollout_log_probs for sample in samples],
        "multimodal_train_inputs": [sample.multimodal_train_inputs for sample in samples],
    }
    (shard,) = split_train_data_by_dp_raw(ARGS, train_data, dp_size=1)
    packed = torch.cat([entry[KEY] for entry in shard["multimodal_train_inputs"]], dim=0)
    # The shape and order the trainer decodes: one row per packed document, document order.
    assert packed.shape == (4, 4)
    for index, tokens in enumerate(shard["tokens"]):
        assert packed[index][0].item() == tokens[-1]
