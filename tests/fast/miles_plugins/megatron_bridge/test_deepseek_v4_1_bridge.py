"""CPU tests for the DeepSeek-V4.1 Megatron-Bridge bridge: its weight names and its FP8 / MXFP4 conversion.

The names must be exactly Miles' mbridge V4.1 tables, split over tensor-parallel ranks the way Miles' layers shard
each weight; quantized weights must load to their dequantized values and export back to them.
"""

from types import SimpleNamespace

import pytest
import torch
from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="stage-a-cpu", labels=[])

pytest.importorskip("mbridge")
pytest.importorskip("megatron.bridge")
pytest.importorskip("transformer_engine")

from megatron.bridge.models.conversion import quantization_utils  # noqa: E402
from megatron.bridge.models.conversion.model_bridge import WeightConversionTask  # noqa: E402
from megatron.bridge.models.conversion.param_mapping import (  # noqa: E402
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.deepseek.deepseek_v4_bridge import _HCAlphaMapping, _HCAlphaSecondaryMapping  # noqa: E402

from miles_plugins.mbridge.deepseek_v4_1 import DeepseekV41Bridge as MBridgeDeepseekV41  # noqa: E402
from miles_plugins.megatron_bridge.deepseek_v4_1 import DeepseekV41Bridge  # noqa: E402

LAYER, EXPERT = 3, 5
FP4_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])
TENSOR_PARALLEL = {
    "self_attention.linear_o_group_proj": ColumnParallelMapping,
    "self_attention.core_attention.attn_sink": ColumnParallelMapping,
    "self_attention.linear_q_up_proj.weight": AutoMapping,
    "self_attention.linear_proj.weight": AutoMapping,
    "mlp.shared_experts.linear_fc2.weight": AutoMapping,
    "mlp.experts.linear_fc2.weight": AutoMapping,
}


def _bridge() -> DeepseekV41Bridge:
    bridge = DeepseekV41Bridge()
    bridge.hf_config = SimpleNamespace(quantization_config={"weight_block_size": [32, 32]})
    return bridge


def _layer_entries():
    for table in (
        MBridgeDeepseekV41._ATTENTION_MAPPING,
        MBridgeDeepseekV41._MLP_MAPPING,
        MBridgeDeepseekV41._OTHER_MAPPING,
    ):
        yield from table.items()


def test_every_mbridge_name_resolves_to_the_same_checkpoint_names():
    registry = _bridge().mapping_registry()
    for megatron, hf in MBridgeDeepseekV41._DIRECT_MAPPING.items():
        assert registry.megatron_to_hf_lookup(megatron).hf_param == hf
    for name, hf_names in _layer_entries():
        expert = "{expert_id}" in hf_names[0]
        mapping = registry.megatron_to_hf_lookup(f"decoder.layers.{LAYER}.{name}{EXPERT if expert else ''}")
        expected = [hf.format(layer_number=LAYER, expert_id=EXPERT) for hf in hf_names]
        if len(expected) == 2:
            assert mapping.hf_param == {"gate": expected[0], "up": expected[1]}, name
        else:
            assert mapping.hf_param == expected[0], name


def test_each_weight_splits_over_tensor_parallel_ranks_as_its_layer_shards_it():
    registry = _bridge().mapping_registry()
    for name, hf_names in _layer_entries():
        expert = "{expert_id}" in hf_names[0]
        mapping = registry.megatron_to_hf_lookup(f"decoder.layers.{LAYER}.{name}{EXPERT if expert else ''}")
        param = name.rpartition(".")[2]
        if param == "alpha_pre":
            expected = _HCAlphaMapping
        elif param in ("alpha_post", "alpha_res"):
            expected = _HCAlphaSecondaryMapping
        elif len(hf_names) == 2:
            expected = GatedMLPMapping
        else:
            expected = TENSOR_PARALLEL.get(name, ReplicatedMapping)
        assert type(mapping) is expected, f"{name}: {type(mapping).__name__}, not {expected.__name__}"


def _dequant_fp8(weight, scale):
    rows, cols = weight.shape
    scale = scale.float().repeat_interleave(32, 0)[:rows].repeat_interleave(32, 1)[:, :cols]
    return weight.float() * scale


def _dequant_fp4(weight, scale):
    packed = weight.view(torch.uint8)
    values = torch.stack([FP4_VALUES[(packed & 0xF).long()], FP4_VALUES[(packed >> 4).long()]], dim=-1).flatten(-2)
    return values * scale.float().repeat_interleave(32, 1)


def _checkpoint():
    """One FP8 attention weight at 32x32 ue8m0 blocks and one MXFP4 routed-expert weight, as V4.1 stores them."""
    generator = torch.Generator().manual_seed(0)
    dense = torch.randn(96, 160, generator=generator)
    expert = torch.randn(64, 128, generator=generator)
    e8m0 = torch.empty(1, dtype=torch.float8_e8m0fnu)
    fp8, fp8_scale = quantization_utils.quantize_fp8_e4m3fn_like_scale(dense, e8m0.expand(3, 5), block_size=32)
    fp4, fp4_scale = quantization_utils.quantize_mxfp4_e2m1_like_scale(expert, e8m0.expand(64, 4))
    return {
        "layers.0.attn.wq_a.weight": fp8,
        "layers.0.attn.wq_a.scale": fp8_scale,
        "layers.0.ffn.experts.7.w1.weight": fp4,
        "layers.0.ffn.experts.7.w1.scale": fp4_scale,
        "layers.0.attn_norm.weight": torch.randn(160, generator=generator).bfloat16(),
    }


def test_quantized_weights_load_dequantized():
    checkpoint = _checkpoint()
    bridge = _bridge()
    dense = bridge.maybe_modify_loaded_hf_weight("layers.0.attn.wq_a.weight", checkpoint).cpu()
    expert = bridge.maybe_modify_loaded_hf_weight("layers.0.ffn.experts.7.w1.weight", checkpoint).cpu()
    norm = bridge.maybe_modify_loaded_hf_weight("layers.0.attn_norm.weight", checkpoint)
    assert torch.equal(
        dense.float(), _dequant_fp8(checkpoint["layers.0.attn.wq_a.weight"], checkpoint["layers.0.attn.wq_a.scale"])
    )
    assert torch.equal(
        expert.float(),
        _dequant_fp4(checkpoint["layers.0.ffn.experts.7.w1.weight"], checkpoint["layers.0.ffn.experts.7.w1.scale"]),
    )
    assert norm is checkpoint["layers.0.attn_norm.weight"]


@pytest.mark.parametrize("name", ["layers.0.attn.wq_a.weight", "layers.0.ffn.experts.7.w1.weight"])
def test_export_quantizes_back_to_the_loaded_values(name):
    checkpoint = _checkpoint()
    bridge = _bridge()
    loaded = bridge.maybe_modify_loaded_hf_weight(name, checkpoint)
    task = WeightConversionTask(param_name="p", global_param_name="p", mapping=None)

    exported = bridge.maybe_modify_converted_hf_weight(task, {name: loaded}, checkpoint)

    scale_name = name.removesuffix(".weight") + ".scale"
    weight, scale = exported[name].cpu(), exported[scale_name].cpu()
    assert weight.dtype == checkpoint[name].dtype and scale.dtype == checkpoint[scale_name].dtype
    dequant = _dequant_fp4 if weight.dtype == torch.int8 else _dequant_fp8
    assert torch.equal(dequant(weight, scale), loaded.float().cpu())


def test_export_with_a_weight_dtype_keeps_the_weights_unquantized():
    checkpoint = _checkpoint()
    bridge = _bridge()
    loaded = bridge.maybe_modify_loaded_hf_weight("layers.0.attn.wq_a.weight", checkpoint)
    task = WeightConversionTask(param_name="p", global_param_name="p", mapping=None, weight_dtype=torch.bfloat16)

    exported = bridge.maybe_modify_converted_hf_weight(task, {"layers.0.attn.wq_a.weight": loaded}, checkpoint)

    assert exported == {"layers.0.attn.wq_a.weight": loaded}


def test_export_returns_a_parameter_kept_in_fp32_in_the_source_dtype():
    checkpoint = _checkpoint()
    bridge = _bridge()
    norm = checkpoint["layers.0.attn_norm.weight"]
    task = WeightConversionTask(param_name="p", global_param_name="p", mapping=None)

    exported = bridge.maybe_modify_converted_hf_weight(task, {"layers.0.attn_norm.weight": norm.float()}, checkpoint)

    assert exported["layers.0.attn_norm.weight"].dtype == torch.bfloat16
    assert torch.equal(exported["layers.0.attn_norm.weight"], norm)
