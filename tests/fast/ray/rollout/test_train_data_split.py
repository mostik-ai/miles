from types import SimpleNamespace

import pytest

from miles.ray.rollout.train_data_conversion import split_train_data_by_dp_raw


@pytest.mark.parametrize("balance_data", [False, True])
def test_each_dp_shard_keeps_the_train_metadata_of_its_own_samples(balance_data):
    lengths = [5, 9, 3, 12, 7, 4]
    data = {
        "tokens": [[row] * length for row, length in enumerate(lengths)],
        "response_lengths": [1] * len(lengths),
        "metadata": [{"row": row} for row in range(len(lengths))],
    }

    shards = split_train_data_by_dp_raw(SimpleNamespace(balance_data=balance_data), data, dp_size=2)

    assert sorted(entry["row"] for shard in shards for entry in shard["metadata"]) == list(range(len(lengths)))
    for shard in shards:
        assert [entry["row"] for entry in shard["metadata"]] == [tokens[0] for tokens in shard["tokens"]]
