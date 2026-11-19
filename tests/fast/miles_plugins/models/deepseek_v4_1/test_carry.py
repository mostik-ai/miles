"""CPU tests for the state DeepSeek-V4.1 layers carry inside their output under pipeline parallelism and recompute.

Every layer packs what later layers read, so a chunk boundary after any layer, with or without virtual stages or
uneven stages, hands the next chunk everything it needs. The tests spell out what each layer reads and writes, one
layer at a time, and check the carry plans against it for the real model and for the cut test model.
"""

from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

from miles_plugins.models.deepseek_v4_1.carry import (
    V41CarryPlan,
    V41Runtime,
    v41_pack_carry,
    v41_pack_pre,
    v41_unpack_carry,
    v41_unpack_pre,
)

register_cpu_ci(est_time=2, suite="stage-a-cpu", labels=[])

FLASH = SimpleNamespace(
    num_layers=40,
    csa_compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
    v41_kv_source_layer_ids=(2, 8, 14, 20),
    v41_index_source_layer_ids=(2, 8, 14, 20, 24, 28, 32, 36),
    v41_candidate_source_layer_id=20,
    v41_candidate_block_size=8,
    v41_engram_layer_ids=(1, 14),
)
CUT = SimpleNamespace(
    num_layers=9,
    csa_compress_ratios=[0, 0, 2, 2, 1, 1, 1, 1, 1],
    v41_kv_source_layer_ids=(2, 4),
    v41_index_source_layer_ids=(2, 4, 6),
    v41_candidate_source_layer_id=4,
    v41_candidate_block_size=8,
    v41_engram_layer_ids=(1,),
)


def _source(config, layer):
    ratio = config.csa_compress_ratios[layer]
    return max(s for s in config.v41_kv_source_layer_ids if s <= layer and config.csa_compress_ratios[s] == ratio)


def _reads_and_writes(config, layer):
    """What one layer reads from earlier layers and writes for later ones, from the model's own rules."""
    reads, writes = {"pre"} if layer else set(), {"pre"}
    if layer in config.v41_engram_layer_ids:
        reads.add("ids")
    if config.csa_compress_ratios[layer]:
        source = _source(config, layer)
        index_source = layer in config.v41_index_source_layer_ids
        if source == layer:
            writes.add(("latent", layer))
            if index_source:
                writes.add(("index_k", layer))
        else:
            reads.add(("latent", source))
            if index_source:
                reads.add(("index_k", source))
        if index_source:
            writes.add("topk")
            if layer == config.v41_candidate_source_layer_id:
                writes.add("candidates")
            elif 0 <= config.v41_candidate_source_layer_id < layer:
                reads.add("candidates")
        else:
            reads.add("topk")
    return reads, writes


def _carried(plan):
    carried = {"pre"} | {("latent", k) for k in plan.latent_keys} | {("index_k", k) for k in plan.index_k_keys}
    carried |= {name for name, kept in (("topk", plan.topk), ("candidates", plan.candidates)) if kept}
    return carried | ({"ids"} if plan.input_ids else set())


@pytest.mark.parametrize("config", [FLASH, CUT], ids=["flash", "cut"])
def test_each_layer_gets_what_it_reads_from_the_carry_of_the_layer_before(config):
    state = {"ids"}
    for layer in range(config.num_layers):
        reads, writes = _reads_and_writes(config, layer)
        assert reads <= state, f"layer {layer} reads {reads - state}, which the carry dropped"
        state |= writes
        carried = _carried(V41CarryPlan(config, layer))
        assert carried <= state, f"layer {layer} would pack {carried - state}, which no layer wrote"
        state = carried


@pytest.mark.parametrize("config", [FLASH, CUT], ids=["flash", "cut"])
def test_carry_holds_only_state_a_later_layer_reads(config):
    for layer in range(config.num_layers):
        later_reads = set().union(
            *(_reads_and_writes(config, after)[0] for after in range(layer + 1, config.num_layers))
        )
        carried = _carried(V41CarryPlan(config, layer)) - {"pre"}
        assert carried <= later_reads, f"layer {layer} carries {carried - later_reads} that no later layer reads"


def test_last_layer_carries_only_the_mixing_weights_the_fold_reads():
    plan = V41CarryPlan(FLASH, FLASH.num_layers - 1)
    assert _carried(plan) == {"pre"}


def test_pre_and_its_gradient_survive_the_carry_exactly():
    generator = torch.Generator().manual_seed(0)
    pre = (torch.rand(7, 2, 4, generator=generator) * 3).requires_grad_()
    upstream = torch.randn(7, 2, 4, generator=generator)
    restored = v41_unpack_pre(v41_pack_pre(pre, torch.bfloat16))
    assert torch.equal(restored, pre)
    (restored * upstream).sum().backward()
    assert not torch.equal(upstream.bfloat16().float(), upstream)
    assert torch.equal(pre.grad, upstream)


def test_pre_gradient_crosses_a_whole_layer_output_exactly():
    s, b, base = 37, 1, 64
    generator = torch.Generator().manual_seed(0)
    rt = _runtime(s, b, with_everything=False)
    rt.pre.requires_grad_()
    plan = SimpleNamespace(input_ids=False, latent_keys=[], index_k_keys=[], topk=False, candidates=False)
    upstream = torch.randn(s, b, 4, generator=generator)
    restored = V41Runtime()
    v41_unpack_carry(v41_pack_carry(torch.randn(s, b, base).bfloat16(), rt, plan), restored, base)
    (restored.pre * upstream).sum().backward()
    assert torch.equal(rt.pre.grad, upstream)


def _runtime(s, b, *, with_everything):
    generator = torch.Generator().manual_seed(0)
    rt = V41Runtime()
    rt.pre = torch.rand(s, b, 4, generator=generator)
    rt.input_ids = torch.randint(0, 129280, (b, s), generator=generator)
    if with_everything:
        rt.latent = {2: torch.randn(b, s // 2, 512, generator=generator).bfloat16()}
        rt.index_k = {2: torch.randn(b, s // 2, 128, generator=generator).bfloat16()}
        rt.topk = torch.randint(-1, s // 2, (b, s, 64), generator=generator)
        rt.candidates = torch.rand(b, s, 9, generator=generator) < 0.5
    return rt


def test_carry_round_trips_every_section():
    s, b, base = 37, 1, 64
    rt = _runtime(s, b, with_everything=True)
    plan = SimpleNamespace(input_ids=True, latent_keys=[2], index_k_keys=[2], topk=True, candidates=True)
    hidden = torch.randn(s, b, base).bfloat16()
    packed = v41_pack_carry(hidden, rt, plan)
    assert packed.shape[:2] == (s, b) and packed.shape[-1] > base
    restored = V41Runtime()
    assert torch.equal(v41_unpack_carry(packed, restored, base), hidden)
    assert torch.equal(restored.pre, rt.pre)
    assert torch.equal(restored.input_ids, rt.input_ids)
    assert torch.equal(restored.latent[2], rt.latent[2])
    assert torch.equal(restored.index_k[2], rt.index_k[2])
    assert torch.equal(restored.topk, rt.topk)
    assert torch.equal(restored.candidates, rt.candidates)


def test_output_without_carry_columns_resets_the_runtime():
    rt = _runtime(5, 1, with_everything=True)
    hidden = torch.randn(5, 1, 16).bfloat16()
    assert v41_unpack_carry(hidden, rt, 16) is hidden
    assert rt.pre is None and rt.topk is None and not rt.latent
