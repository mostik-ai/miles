"""Megatron-Bridge support for DeepSeek-V4.1 Flash (``DeepseekV41ForCausalLM``, ``model_type: deepseek_v41``).

The bridge builds Miles' V4.1 layers (``miles_plugins.models.deepseek_v4_1``): they have tensor parallelism, the
Engram, the source layers' shared latents and index keys, and the candidate blocks, which Megatron's own DeepSeek-V4
attention lacks. The model config is the one Miles trains V4.1 with (``scripts/models/deepseek-v4.1.py``), read from
the Hugging Face config; the weight names are Miles' mbridge V4.1 tables.

The served checkpoint keeps dense weights in FP8 with 32x32 ue8m0 block scales and the routed experts in MXFP4. The
bridge dequantizes both while it loads and quantizes them back to the source layout on export, with the
Megatron-Bridge fork's quantization helpers. An export that names a ``weight_dtype``, as Miles' weight sync does,
stays in that dtype.

Not built: the MTP and DSpark layers (``mtp.*``), the vision tower and its aligner. The Engram embedding tables are
not Megatron parameters: Miles' Engram reads its rows from the checkpoint itself (``v41_hf_checkpoint``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import torch
from megatron.bridge.models.conversion import quantization_utils
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge, WeightConversionTask
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    ReplicatedMapping,
)
from megatron.bridge.models.conversion.transformers_compat import rope_theta_from_hf
from megatron.bridge.models.deepseek.deepseek_v4_bridge import _HCAlphaMapping, _HCAlphaSecondaryMapping
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.moe.router import TopKRouter

from miles_plugins.mbridge.deepseek_v4_1 import DeepseekV41Bridge as MBridgeDeepseekV41
from miles_plugins.models.deepseek_v4_1.deepseek_v4_1 import apply_v41_config, dsv41_block_spec

# Bare parameters split over tensor-parallel ranks along their first dimension.
_COLUMN_PARALLEL_PARAMS = ("self_attention.linear_o_group_proj", "self_attention.core_attention.attn_sink")
# Weights of Megatron's tensor-parallel linears, whose split AutoMapping reads from the module.
_PARALLEL_LINEAR_WEIGHTS = (
    "self_attention.linear_q_up_proj.weight",
    "self_attention.linear_proj.weight",
    "mlp.shared_experts.linear_fc2.weight",
    "mlp.experts.linear_fc2.weight",
)
_HC_ALPHAS = ("alpha_pre", "alpha_post", "alpha_res")


@dataclass
class DeepseekV41ModelProvider(MLAModelProvider):
    """The MLA provider plus the fields Miles' V4.1 layers read; each chunk's layers come from ``dsv41_block_spec``.

    ``v41_hf_checkpoint`` is where Engram reads its tables and its tokenizer. The ``v41_*`` layer fields are
    ``apply_v41_config``'s, from the Hugging Face config.
    """

    transformer_layer_spec: ModuleSpec | Callable = field(default_factory=lambda: dsv41_block_spec)
    # Each layer appends the state later layers need to its output, so pipeline stages exchange variable widths.
    variable_seq_lengths: bool = True
    v41_hf_checkpoint: str | None = None
    miles_dsa_topk_backend: str = "canonical"
    v41_kv_source_layer_ids: tuple[int, ...] = ()
    v41_index_source_layer_ids: tuple[int, ...] = ()
    v41_candidate_source_layer_id: int = -1
    v41_candidate_topk_blocks: int = 0
    v41_candidate_block_size: int = 0
    v41_engram_layer_ids: tuple[int, ...] = ()
    v41_engram_num_embeddings: tuple[int, ...] = ()
    v41_engram_max_ngram_size: int = 1
    v41_engram_vocab_size: int = 0
    v41_engram_n_heads: int = 0
    v41_engram_head_dim: int = 0
    v41_engram_pad_token_id: int = 2
    v41_engram_compressed_vocab_size: int = 0


@MegatronModelBridge.register_bridge(
    source="DeepseekV41ForCausalLM", target=GPTModel, provider=DeepseekV41ModelProvider, model_type="deepseek_v41"
)
class DeepseekV41Bridge(MegatronModelBridge):
    """Hugging Face ``DeepseekV41ForCausalLM`` (FP8 + MXFP4, or BF16) <-> Megatron GPTModel on Miles' V4.1 layers."""

    MODEL_CONFIG_CLASS = None

    def provider_bridge(self, hf_pretrained) -> DeepseekV41ModelProvider:
        hf = hf_pretrained.config
        provider = super().provider_bridge(hf_pretrained)
        num_layers = hf.num_hidden_layers

        provider.num_layers = num_layers
        provider.num_query_groups = hf.num_attention_heads
        provider.ffn_hidden_size = hf.moe_intermediate_size
        provider.normalization = "RMSNorm"
        provider.layernorm_epsilon = hf.rms_norm_eps
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.add_qkv_bias = False
        provider.share_embeddings_and_output_weights = False
        provider.hidden_dropout = 0.0
        provider.attention_dropout = 0.0
        provider.attention_softmax_in_fp32 = True

        provider.multi_latent_attention = True
        provider.experimental_attention_variant = "dsv4"
        provider.q_lora_rank = hf.q_lora_rank
        provider.kv_lora_rank = hf.head_dim
        provider.qk_head_dim = hf.head_dim
        provider.v_head_dim = hf.head_dim
        provider.qk_pos_emb_head_dim = hf.qk_rope_head_dim
        provider.qk_layernorm = True
        provider.o_groups = hf.o_groups
        provider.o_lora_rank = hf.o_lora_rank
        provider.csa_window_size = hf.sliding_window
        # The checkpoint lists the MTP layers' ratios after the decoder's; only the decoder is built.
        provider.csa_compress_ratios = list(hf.compress_ratios[:num_layers])
        provider.csa_compress_rotary_base = hf.compress_rope_theta
        provider.dsa_indexer_n_heads = hf.index_n_heads
        provider.dsa_indexer_head_dim = hf.index_head_dim
        provider.dsa_indexer_topk = hf.index_topk

        provider.rotary_base = rope_theta_from_hf(hf)
        provider.rotary_scaling_factor = hf.rope_scaling["factor"]
        provider.original_max_position_embeddings = hf.rope_scaling["original_max_position_embeddings"]
        provider.beta_fast = hf.rope_scaling["beta_fast"]
        provider.beta_slow = hf.rope_scaling["beta_slow"]
        provider.apply_rope_fusion = False

        provider.enable_hyper_connections = True
        provider.num_residual_streams = hf.hc_mult
        provider.mhc_sinkhorn_iterations = hf.hc_sinkhorn_iters

        provider.num_moe_experts = hf.n_routed_experts
        provider.moe_layer_freq = [1] * num_layers
        provider.moe_ffn_hidden_size = hf.moe_intermediate_size
        provider.moe_shared_expert_intermediate_size = hf.moe_intermediate_size * hf.n_shared_experts
        provider.moe_router_topk = hf.num_experts_per_tok
        provider.moe_router_topk_scaling_factor = hf.routed_scaling_factor
        provider.moe_router_score_function = hf.scoring_func
        provider.moe_router_pre_softmax = True
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_bias_update_rate = 0.0
        # The DeepSeek-V3 config class Miles reads V4.1 with defaults to grouped routing, which V4.1 does not have.
        provider.moe_router_num_groups = None
        provider.moe_router_group_topk = None
        # as Miles trains every model (set_default_megatron_args): the fp32 torch GEMM keeps the routing logits
        # closer to the rollout's than TE's bf16 one
        provider.moe_router_use_torch_mm = True
        provider.moe_router_load_balancing_type = "none"
        provider.moe_aux_loss_coeff = 0.0
        provider.moe_n_hash_layers = 0
        provider.moe_grouped_gemm = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.activation_func_clamp_value = hf.swiglu_limit
        provider.activation_func_clamp_shared_expert = True
        provider.bias_activation_fusion = False

        provider.mtp_num_layers = None
        provider.v41_hf_checkpoint = str(hf_pretrained.model_name_or_path)
        apply_v41_config(provider, hf)
        return provider

    def load_weights_hf_to_megatron(self, hf_pretrained, megatron_model, allowed_mismatched_params=None):
        """Load as the base bridge does, into the routers' expert bias in fp32.

        Float16Module casts every buffer to bf16; Megatron's router restores its expert bias to fp32 only when it
        runs, so a load into the bf16 buffer would round the checkpoint's fp32 bias.
        """
        for chunk in megatron_model if isinstance(megatron_model, list) else [megatron_model]:
            for module in chunk.modules():
                if isinstance(module, TopKRouter):
                    module._maintain_float32_expert_bias()
        return super().load_weights_hf_to_megatron(hf_pretrained, megatron_model, allowed_mismatched_params)

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Megatron-Bridge mappings for the names of Miles' mbridge V4.1 tables."""
        mappings: list[MegatronParamMapping] = [
            AutoMapping(megatron, hf) for megatron, hf in MBridgeDeepseekV41._DIRECT_MAPPING.items()
        ]
        layer_tables = (
            MBridgeDeepseekV41._ATTENTION_MAPPING,
            MBridgeDeepseekV41._MLP_MAPPING,
            MBridgeDeepseekV41._OTHER_MAPPING,
        )
        for table in layer_tables:
            mappings += [_layer_mapping(name, hf_names) for name, hf_names in table.items()]
        return MegatronMappingRegistry(*mappings)

    def maybe_modify_loaded_hf_weight(
        self, hf_param: str | dict[str, str], hf_state_dict: Mapping[str, torch.Tensor]
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Dequantize FP8 block-scaled and MXFP4 weights to BF16; other tensors load as stored."""
        if isinstance(hf_param, dict):
            return {role: self._load_hf_weight(name, hf_state_dict) for role, name in hf_param.items()}
        return self._load_hf_weight(hf_param, hf_state_dict)

    def maybe_modify_converted_hf_weight(
        self,
        task: WeightConversionTask,
        converted_weights_dict: dict[str, torch.Tensor],
        hf_state_dict: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Quantize the weights the source stores quantized back to its layout, unless the export names a dtype."""
        if task.weight_dtype is not None:
            return converted_weights_dict
        result = {}
        for name, weight in converted_weights_dict.items():
            scale_name = _scale_name(name)
            if scale_name is not None and scale_name in hf_state_dict:
                result[name], result[scale_name] = self._quantize_like_source(name, weight, hf_state_dict[scale_name])
            elif weight.dtype == torch.float32 and name in hf_state_dict:
                # parameters Miles keeps in fp32 (norms, Engram's q/k weights) go back in the source's dtype
                result[name] = weight.to(hf_state_dict[name].dtype)
            else:
                result[name] = weight
        return result

    def _load_hf_weight(self, name: str, hf_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        weight = hf_state_dict[name]
        scale_name = _scale_name(name)
        if scale_name is None or scale_name not in hf_state_dict:
            return weight
        # every rank dequantizes what it loads; on the host the ranks of a node contend for its cores
        device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        weight, scale = weight.to(device), hf_state_dict[scale_name].to(device)
        if weight.dtype == torch.int8:
            return quantization_utils.dequantize_mxfp4_e2m1_packed(weight, scale)
        if weight.dtype == torch.float8_e4m3fn:
            return quantization_utils.dequantize_fp8_e4m3fn_with_scale(
                weight, scale, name=name, block_size=self._fp8_block_size
            )
        raise ValueError(f"{name}: a {weight.dtype} weight with a scale is neither FP8 nor MXFP4")

    def _quantize_like_source(
        self, name: str, weight: torch.Tensor, source_scale: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if ".ffn.experts." in name and quantization_utils.is_mxfp4_e2m1_scale_geometry(weight, source_scale):
            return quantization_utils.quantize_mxfp4_e2m1_like_scale(weight, source_scale, name=name)
        return quantization_utils.quantize_fp8_e4m3fn_like_scale(
            weight, source_scale, name=name, block_size=self._fp8_block_size
        )

    @property
    def _fp8_block_size(self) -> int:
        rows, cols = self.hf_config.quantization_config["weight_block_size"]
        assert rows == cols, f"DeepSeek-V4.1 FP8 blocks are square, got {rows}x{cols}"
        return rows


def _layer_mapping(name: str, hf_names: list[str]) -> MegatronParamMapping:
    """The Megatron-Bridge mapping of one entry of an mbridge V4.1 per-layer table."""
    hf = [hf_name.replace("{layer_number}", "*").replace("{expert_id}", "*") for hf_name in hf_names]
    megatron = f"decoder.layers.*.{name}"
    if "{expert_id}" in hf_names[0]:
        # grouped experts keep one weight per local expert: linear_fc1.weight0, weight1, ...
        megatron += "*"
    module, _, param = name.rpartition(".")
    if param in _HC_ALPHAS:
        return _hc_alpha_mapping(module, param, hf[0])
    if len(hf) == 2:
        return GatedMLPMapping(megatron, gate=hf[0], up=hf[1])
    if name in _COLUMN_PARALLEL_PARAMS:
        return ColumnParallelMapping(megatron, hf[0])
    if name in _PARALLEL_LINEAR_WEIGHTS:
        return AutoMapping(megatron, hf[0])
    return ReplicatedMapping(megatron, hf[0])


def _hc_alpha_mapping(module: str, alpha: str, hf_scale: str) -> MegatronParamMapping:
    """The checkpoint packs a hyper-connection's three alphas as one ``[pre, post, res]`` tensor."""
    names = {name: f"decoder.layers.*.{module}.{name}" for name in _HC_ALPHAS}
    if alpha == "alpha_pre":
        return _HCAlphaMapping(names["alpha_pre"], names["alpha_post"], names["alpha_res"], hf_scale)
    return _HCAlphaSecondaryMapping(names[alpha], hf_scale, _HC_ALPHAS.index(alpha))


def _scale_name(name: str) -> str | None:
    return name.removesuffix(".weight") + ".scale" if name.endswith(".weight") else None
