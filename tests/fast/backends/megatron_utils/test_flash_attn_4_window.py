import pytest
import torch
import torch.nn.functional as F

from miles.backends.megatron_utils.flash_attn_4_window import patch_flash_attn_4_window, unbounded_as_none


def test_unbounded_window_sides_become_none_and_bounded_sides_stay():
    assert unbounded_as_none((-1, 0)) == (None, 0)
    assert unbounded_as_none((-1, -1)) == (None, None)
    assert unbounded_as_none((128, 0)) == (128, 0)
    assert unbounded_as_none(None) is None


def causal_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, cu_seqlens: list[int]) -> torch.Tensor:
    repeat = q.shape[1] // k.shape[1]
    outputs = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        query, key, value = (tensor[start:end].transpose(0, 1) for tensor in (q, k, v))
        key, value = key.repeat_interleave(repeat, 0), value.repeat_interleave(repeat, 0)
        outputs.append(F.scaled_dot_product_attention(query, key, value, is_causal=True).transpose(0, 1))
    return torch.cat(outputs)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual.float() - expected).norm() / expected.norm())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlashAttention 4 runs on a GPU")
@pytest.mark.parametrize("head_dim", [128, 256])
def test_transformer_engine_causal_window_runs_causal_attention_on_flash_attn_4(head_dim: int):
    pytest.importorskip("flash_attn.cute")
    backends = pytest.importorskip("transformer_engine.pytorch.attention.dot_product_attention.backends")
    patch_flash_attn_4_window()
    torch.manual_seed(0)
    cu_seqlens = [0, 96, 256]
    leaves = [
        torch.randn(256, heads, head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for heads in (8, 2, 2)
    ]
    references = [leaf.detach().float().requires_grad_() for leaf in leaves]

    output = backends.flash_attn_varlen_func_v4(
        *leaves,
        cu_seqlens_q=torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda"),
        cu_seqlens_k=torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda"),
        max_seqlen_q=160,
        max_seqlen_k=160,
        softmax_scale=head_dim**-0.5,
        causal=True,
        window_size=(-1, 0),
    )
    output = output[0] if isinstance(output, tuple) else output
    expected = causal_reference(*references, cu_seqlens)
    output_grad = torch.randn_like(expected)
    output.backward(output_grad.to(output.dtype))
    expected.backward(output_grad)

    assert relative_error(output, expected) < 2e-2
    for leaf, reference in zip(leaves, references, strict=True):
        assert relative_error(leaf.grad, reference.grad) < 2e-2
