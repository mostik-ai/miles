from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="stage-c-4-gpu-b200", labels=["megatron"], hardware=["blackwell"], num_gpus=2)


import functools
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tests.fast.dist_utils import run_multiprocess

from miles.backends.megatron_utils.flash_attn_4_ulysses import patch_flash_attn_4_ulysses

HEADS, KV_HEADS, HEAD_DIM = 8, 2, 256
LENGTHS = [1024, 512, 1536]


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


def _counting_flash_attn_4_kernels() -> dict[str, int]:
    from flash_attn.cute import interface

    calls = {"_flash_attn_fwd": 0, "_flash_attn_bwd": 0}
    for name in calls:
        kernel = getattr(interface, name)

        # the kernels read their compile caches as attributes of their own module-level names
        @functools.wraps(kernel)
        def counted(*args, _kernel=kernel, _name=name, **kwargs):
            calls[_name] += 1
            return _kernel(*args, **kwargs)

        setattr(interface, name, counted)
    return calls


def _check_ulysses_attention(rank: int, world_size: int, port: int) -> None:
    # attention_backend "flash", as Megatron sets it, so Transformer Engine does not pick its fused backend
    os.environ["NVTE_FUSED_ATTN"], os.environ["NVTE_UNFUSED_ATTN"] = "0", "0"
    os.environ["MASTER_ADDR"], os.environ["MASTER_PORT"] = "localhost", str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        # transformer_engine_torch is registered when transformer_engine.pytorch loads, so it is imported after it
        from transformer_engine.pytorch import DotProductAttention
        import transformer_engine_torch as tex  # isort: skip

        calls = _counting_flash_attn_4_kernels()
        patch_flash_attn_4_ulysses()
        torch.manual_seed(0)
        cu_seqlens = [0, *torch.tensor(LENGTHS).cumsum(0).tolist()]
        cu = torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda")
        total = cu_seqlens[-1]
        q, k, v, output_grad = (
            torch.randn(total, heads, HEAD_DIM, device="cuda") for heads in (HEADS, KV_HEADS, KV_HEADS, HEADS)
        )
        mine = tex.thd_get_partitioned_indices(cu, total, world_size, rank)
        shards = [tensor[mine].bfloat16().requires_grad_() for tensor in (q, k, v)]
        references = [tensor.requires_grad_() for tensor in (q, k, v)]
        attention = DotProductAttention(
            HEADS, HEAD_DIM, num_gqa_groups=KV_HEADS, attn_mask_type="padding_causal", qkv_format="thd"
        )
        attention.set_context_parallel_group(dist.group.WORLD, list(range(world_size)), torch.cuda.Stream(), "a2a")

        output = attention(
            *shards, cu_seqlens_q=cu, cu_seqlens_kv=cu, max_seqlen_q=max(LENGTHS), max_seqlen_kv=max(LENGTHS)
        ).reshape(-1, HEADS, HEAD_DIM)
        output.backward(output_grad[mine].bfloat16())
        expected = causal_reference(*references, cu_seqlens)
        expected.backward(output_grad)

        assert calls["_flash_attn_fwd"] > 0 and calls["_flash_attn_bwd"] > 0
        assert relative_error(output, expected[mine].detach()) < 2e-2
        for shard, reference in zip(shards, references, strict=True):
            assert relative_error(shard.grad, reference.grad[mine]) < 2e-2
    finally:
        dist.destroy_process_group()


def test_ulysses_attention_runs_on_flash_attn_4_and_matches_float32_attention_and_its_gradients():
    pytest.importorskip("flash_attn.cute")
    run_multiprocess(_check_ulysses_attention, world_size=2)
