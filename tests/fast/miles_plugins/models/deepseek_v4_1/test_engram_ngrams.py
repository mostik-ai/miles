"""CPU tests for DeepSeek-V4.1 Engram's n-gram hashing on packed (thd) streams.

An n-gram ending near the start of a packed sequence must not reach into the sequence before it: it pads, as it does
when the sequence runs alone.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles_plugins.models.deepseek_v4.ops.thd_utils import get_q_positions_thd

register_cpu_ci(est_time=2, suite="stage-a-cpu", labels=[])

ROOT = Path(__file__).resolve().parents[5]
MAX_NGRAM = 4
LENS = [9, 2, 6]


@pytest.fixture
def engram(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """engram.py loaded without Transformer Engine, which its hashing does not use."""
    stub = ModuleType("megatron.core.extensions.transformer_engine")
    stub.TELinear = object
    monkeypatch.setitem(sys.modules, "megatron.core.extensions.transformer_engine", stub)
    spec = importlib.util.spec_from_file_location(
        "_test_dsv41_engram", ROOT / "miles_plugins/models/deepseek_v4_1/engram.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _hash_ids(engram: ModuleType):
    primes = torch.tensor([[1009, 1013], [1019, 1021], [1031, 1033]])
    hasher = SimpleNamespace(
        token_map=torch.arange(200) % 97,
        multipliers=engram.compute_hash_multipliers([1], MAX_NGRAM, 97)[0],
        primes=primes,
        offsets=torch.cat([torch.zeros(1, dtype=torch.long), primes.flatten()[:-1].cumsum(0)]),
        pad_id=5,
        max_ngram_size=MAX_NGRAM,
    )
    return lambda ids, positions=None: engram.DeepSeekV41Engram.hash_ids(hasher, ids, positions)


def test_packed_ngrams_match_each_sequence_alone(engram):
    hash_ids = _hash_ids(engram)
    sequences = [torch.randint(0, 200, (1, n), generator=torch.Generator().manual_seed(n)) for n in LENS]
    cu = torch.tensor([0, *torch.tensor(LENS).cumsum(0).tolist()], dtype=torch.int32)
    positions = get_q_positions_thd(cu, sum(LENS)).unsqueeze(0)

    packed = hash_ids(torch.cat(sequences, dim=1), positions)

    assert torch.equal(packed, torch.cat([hash_ids(s) for s in sequences], dim=1))


def test_stream_positions_would_hash_across_sequence_edges(engram):
    hash_ids = _hash_ids(engram)
    sequences = [torch.randint(0, 200, (1, n), generator=torch.Generator().manual_seed(n)) for n in LENS]

    unpacked = hash_ids(torch.cat(sequences, dim=1))

    second_start = LENS[0]
    assert not torch.equal(unpacked[:, second_start], hash_ids(sequences[1])[:, 0])
