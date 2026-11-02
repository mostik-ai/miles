from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.model_provider import _apply_bridge_runtime_config


def _runtime_args(**overrides):
    values = {
        "tensor_model_parallel_size": 1,
        "pipeline_model_parallel_size": 8,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "sequence_parallel": False,
        "context_parallel_size": 1,
        "cp_comm_type": ["p2p"],
        "calculate_per_token_loss": False,
        "variable_seq_lengths": True,
        "attention_softmax_in_fp32": True,
        "gradient_accumulation_fusion": True,
        "fp32_residual_connection": False,
        "deterministic_mode": True,
        "recompute_granularity": "full",
        "recompute_method": "uniform",
        "recompute_num_layers": 1,
        "recompute_modules": None,
        "cpu_offloading_num_layers": 0,
        "distribute_saved_activations": False,
        "tp_comm_overlap": False,
        "fp8": "e4m3",
        "fp8_recipe": "mxfp8",
        "attention_backend": "auto",
        "dsa_kernel_backend": "cudnn",
        "mtp_num_layers": None,
        "mtp_detach_heads": True,
        "moe_token_dispatcher_type": "alltoall",
        "decoder_first_pipeline_num_layers": 4,
        "decoder_last_pipeline_num_layers": 3,
        "moe_router_bias_update_rate": None,
        "moe_aux_loss_coeff": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _provider():
    return SimpleNamespace(
        dsa_kernel_backend="none",
        mtp_num_layers=1,
        experimental_attention_variant="dsv4_hybrid",
    )


@pytest.mark.parametrize(
    ("cp_comm_type", "expected"), [(["a2a"], "a2a"), (["a2a", "p2p"], ["a2a", "p2p"])], ids=["one", "per-layer"]
)
def test_bridge_runtime_carries_the_context_parallel_communication_type(cp_comm_type, expected):
    provider = _provider()

    _apply_bridge_runtime_config(provider, _runtime_args(context_parallel_size=4, cp_comm_type=cp_comm_type))

    assert (provider.context_parallel_size, provider.cp_comm_type) == (4, expected)


def test_bridge_runtime_preserves_native_variant_and_disables_unrequested_mtp():
    provider = _provider()

    _apply_bridge_runtime_config(provider, _runtime_args())

    assert provider.experimental_attention_variant == "dsv4_hybrid"
    assert provider.dsa_kernel_backend == "cudnn"
    assert provider.mtp_num_layers is None


@pytest.mark.parametrize("backend", [None, "none", "tilelang", "cudnn"])
def test_bridge_runtime_resolves_native_dsv4_kernel_default(backend):
    provider = _provider()

    _apply_bridge_runtime_config(provider, _runtime_args(dsa_kernel_backend=backend, dsa_attention_backend="tilelang"))

    assert provider.dsa_kernel_backend == (backend or "cudnn")
