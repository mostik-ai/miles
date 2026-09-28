"""Unit tests for the opt-in trainable parameter scope of the base-weight stream.

Covers the rank agreement that turns per-rank ``requires_grad`` into one selection, and the
late unit filter that the backend-neutral iterator applies after atomic-group assembly.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from miles.backends.training_utils.weight_update.hf_weight_iterator import (
    HfWeightIteratorBase,
    WeightUpdatePlacement,
)
from miles.backends.training_utils.weight_update.hf_weight_iterator.bucketing import AtomicUpdateGroup
from miles.backends.training_utils.weight_update.hf_weight_iterator.trainable_scope import (
    local_trainable_global_names,
    trainable_source_names,
)


def _task(param_name, global_param_name=None, *, requires_grad=None):
    """A conversion task: ``requires_grad=None`` is the name-only stand-in another rank owns."""
    weight = None if requires_grad is None else SimpleNamespace(requires_grad=requires_grad)
    return SimpleNamespace(
        param_name=param_name,
        global_param_name=global_param_name or param_name,
        param_weight=weight,
    )


class TestRankAgreement:
    def test_only_owned_trainable_parameters_are_contributed(self):
        tasks = [
            _task("channel.gate", requires_grad=True),
            _task("decoder.layers.0.mlp.weight", requires_grad=False),
            _task("decoder.layers.9.mlp.weight"),
        ]
        assert local_trainable_global_names(tasks) == {"channel.gate"}

    def test_pipeline_ranks_select_the_same_exported_weights(self):
        """Every rank holds one task per global parameter, named locally for the ones it owns.
        Stage 1 even reuses stage 0's layer-0 name for its own layer-1 parameter; the union of
        global names must still select the same two exported weights on both stages."""
        stage0 = [
            _task("channel.gate", requires_grad=True),
            _task("decoder.layers.0.mlp.weight", requires_grad=False),
            _task("decoder.layers.1.mlp.weight"),
            _task("decoder.layers.1.self_attn.weight"),
        ]
        stage1 = [
            _task("channel.gate"),
            _task("decoder.layers.0.mlp.weight"),
            _task("decoder.layers.0.mlp.weight", "decoder.layers.1.mlp.weight", requires_grad=False),
            _task("decoder.layers.0.self_attn.weight", "decoder.layers.1.self_attn.weight", requires_grad=True),
        ]
        agreed = local_trainable_global_names(stage0) | local_trainable_global_names(stage1)

        assert trainable_source_names(stage0, agreed) == {"channel.gate", "decoder.layers.1.self_attn.weight"}
        assert trainable_source_names(stage1, agreed) == {"channel.gate", "decoder.layers.0.self_attn.weight"}

    def test_a_source_name_covering_both_a_trainable_and_a_frozen_parameter_raises(self):
        """One exported name cannot select two parameters that disagree: dropping or keeping it
        would differ from the rank that owns the other one."""
        tasks = [
            _task("decoder.layers.0.mlp.weight", requires_grad=True),
            _task("decoder.layers.0.mlp.weight", "decoder.layers.4.mlp.weight", requires_grad=False),
        ]
        agreed = local_trainable_global_names(tasks)

        with pytest.raises(RuntimeError, match="cannot resolve this parallel layout"):
            trainable_source_names(tasks, agreed)

    def test_a_fully_frozen_rank_contributes_nothing(self):
        assert local_trainable_global_names([_task("decoder.weight", requires_grad=False)]) == set()


class _StubIterator(HfWeightIteratorBase):
    """A backend that reports which of its units came from a trainable parameter."""

    def __init__(self, units, trainable, *, parameter_scope="all", atomic_groups=()):
        super().__init__(
            Namespace(update_weight_buffer_size=1 << 30),
            [],
            placement=WeightUpdatePlacement(gather_pp=True),
            model_name="stub",
            quantization_config=None,
            parameter_scope=parameter_scope,
        )
        self._units = units
        self._trainable = trainable
        self._atomic_groups = list(atomic_groups)

    def _hf_atomic_update_groups(self):
        return self._atomic_groups

    def _iter_hf_param_units(self, weights, *, materialize):
        if not materialize:
            return  # a joining rank drives the same collectives and yields nothing
        for unit in self._units:
            if any(name in self._trainable for name, _tensor in unit):
                self.trainable_hf_names.update(name for name, _tensor in unit)
            yield unit

    def _iter_hf_adapter_units(self, adapter, *, materialize):
        return iter(())


def _unit(*names):
    return [(name, torch.zeros(2)) for name in names]


def _names(buckets):
    return [name for bucket in buckets for name, _tensor in bucket]


def _trainable_iterator(units, trainable, **kwargs):
    return _StubIterator(units, trainable, parameter_scope="trainable", **kwargs)


class TestTrainableScopeFilter:
    def test_the_default_scope_sends_every_unit(self):
        iterator = _StubIterator([_unit("frozen.weight"), _unit("channel.weight")], {"channel.weight"})
        assert iterator.parameter_scope == "all"
        assert _names(iterator.iter_hf_weights(None)) == ["frozen.weight", "channel.weight"]

    def test_trainable_drops_the_frozen_units(self):
        """A unit with no trainable source — a frozen parameter, or a checkpoint passthrough
        tensor with no trainer-side parameter at all — does not reach a bucket."""
        iterator = _trainable_iterator([_unit("frozen.weight"), _unit("channel.weight")], {"channel.weight"})
        assert _names(iterator.iter_hf_weights(None)) == ["channel.weight"]

    def test_a_parameter_frozen_after_a_sync_stops_being_selected(self):
        """Each export re-reports its trainable names; a stale annotation would keep sending a
        weight the trainer no longer trains."""
        iterator = _trainable_iterator(
            [_unit("channel.weight"), _unit("head.weight")], {"channel.weight", "head.weight"}
        )
        assert _names(iterator.iter_hf_weights(None)) == ["channel.weight", "head.weight"]

        iterator._trainable = {"channel.weight"}
        assert _names(iterator.iter_hf_weights(None)) == ["channel.weight"]

    def test_a_unit_survives_whole_when_one_tensor_is_trainable(self):
        """A quantized weight and its scale companion are one indivisible unit."""
        iterator = _trainable_iterator(
            [_unit("channel.weight", "channel.weight_scale"), _unit("frozen.weight", "frozen.weight_scale")],
            {"channel.weight"},
        )
        assert _names(iterator.iter_hf_weights(None)) == ["channel.weight", "channel.weight_scale"]

    def test_an_atomic_group_is_selected_after_assembly(self):
        """The group's members are separate units upstream; dropping one would leave the
        assembler with an incomplete group."""
        iterator = _trainable_iterator(
            [_unit("block.q_weight"), _unit("frozen.weight"), _unit("block.q_scale")],
            {"block.q_weight"},
            atomic_groups=[AtomicUpdateGroup(key="q", suffixes=(".q_weight", ".q_scale"))],
        )
        assert _names(iterator.iter_hf_weights(None)) == ["block.q_weight", "block.q_scale"]

    def test_selecting_nothing_fails_instead_of_syncing_nothing(self):
        iterator = _trainable_iterator([_unit("frozen.weight")], set())
        with pytest.raises(RuntimeError, match="selected no weights"):
            list(iterator.iter_hf_weights(None))

    def test_a_joining_rank_yields_nothing_without_failing(self):
        """materialize=False ranks only join the collectives; they select nothing by design."""
        iterator = _trainable_iterator([_unit("frozen.weight")], set())
        assert list(iterator.iter_hf_weights(None, materialize=False)) == []

    def test_an_unknown_scope_is_rejected(self):
        with pytest.raises(AssertionError, match="Unknown parameter scope"):
            _StubIterator([], set(), parameter_scope="requires_grad")
