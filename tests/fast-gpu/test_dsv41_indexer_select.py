"""DeepSeek-V4.1 indexer_select with the canonical top-k picks the same keys per query as the torch top-k, and a
packed (thd) stream picks for each sequence what that sequence picks alone."""

import sys

import pytest
import torch
from tests.ci.ci_register import register_cuda_ci, register_rocm_ci

register_cuda_ci(est_time=100, suite="stage-b-2-gpu-h200", labels=["precision"], hardware=["hopper", "blackwell"])
register_rocm_ci(est_time=30, suite="nightly-stage-c-2-gpu-mi350", labels=["precision"])

pytest.importorskip("tilelang")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from miles.kernels.attention.dsa.topk import get_dsa_topk_fn  # noqa: E402
from miles_plugins.models.deepseek_v4.ops.thd_utils import ThdLayout  # noqa: E402
from miles_plugins.models.deepseek_v4_1.ops.indexer import indexer_select, indexer_select_packed  # noqa: E402
from miles_plugins.models.deepseek_v4_1.ops.packed import packed_compression  # noqa: E402

HEADS, DIM, TOPK = 32, 128, 512
BLOCK = 128


def _select(q, index_k, weights, compress_lens, candidates, topk_backend, *, is_candidate_source, query_chunk):
    canonical = topk_backend == "canonical"
    return indexer_select(
        q,
        index_k,
        weights,
        compress_lens,
        candidates,
        is_candidate_source=is_candidate_source,
        uses_candidates=candidates is not None,
        candidate_topk_blocks=4,
        candidate_block_size=BLOCK,
        topk=TOPK,
        topk_fn=get_dsa_topk_fn(topk_backend),
        canonical_topk=canonical,
        allow_deep_select=False,
        query_chunk=query_chunk,
    )


@pytest.mark.parametrize("bsz,seqlen,ratio", [(1, 4096, 2), (2, 3001, 1), (1, 1500, 4)])
@pytest.mark.parametrize("query_chunk", [None, 1024])
@pytest.mark.parametrize("candidates", ["none", "source", "user"])
def test_canonical_select_matches_torch_select(bsz, seqlen, ratio, query_chunk, candidates):
    torch.manual_seed(0)
    q = torch.randn(bsz, seqlen, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    n_kv = seqlen // ratio
    index_k = torch.randn(bsz, n_kv, DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(bsz, seqlen, HEADS, device="cuda") * HEADS**-0.5 * DIM**-0.5
    compress_lens = (torch.arange(seqlen, device="cuda") + 1) // ratio
    candidate_mask = None
    if candidates == "user":
        candidate_mask = torch.rand(bsz, seqlen, -(-n_kv // BLOCK), device="cuda") < 0.3
    options = dict(is_candidate_source=candidates == "source", query_chunk=query_chunk)
    torch_idx, torch_cand = _select(q, index_k, weights, compress_lens, candidate_mask, "torch", **options)
    canonical_idx, canonical_cand = _select(q, index_k, weights, compress_lens, candidate_mask, "canonical", **options)

    assert canonical_idx.dtype == torch_idx.dtype == torch.int64
    assert torch.equal(canonical_idx.sort(dim=-1).values, torch_idx)
    valid = canonical_idx >= 0
    assert torch.equal(valid.int(), valid.int().sort(dim=-1, descending=True).values)
    assert bool((canonical_idx < compress_lens.unsqueeze(-1)).all())
    if candidates == "source":
        assert torch.equal(canonical_cand, torch_cand)


def _options(topk_backend, candidates):
    return dict(
        is_candidate_source=candidates == "source",
        uses_candidates=candidates == "user",
        candidate_topk_blocks=4,
        candidate_block_size=BLOCK,
        topk=TOPK,
        topk_fn=get_dsa_topk_fn(topk_backend),
        canonical_topk=topk_backend == "canonical",
        allow_deep_select=False,
        query_chunk=None,
    )


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("topk_backend", ["torch", "canonical"])
@pytest.mark.parametrize("candidates", ["none", "source", "user"])
def test_packed_select_matches_each_sequence_alone(ratio, topk_backend, candidates):
    torch.manual_seed(0)
    lens = [1401, 33, 2203]
    total = sum(lens)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32, device="cuda")
    segments = packed_compression(
        ThdLayout(cu_seqlens=cu, seq_lens=tuple(lens), global_start=0, max_seqlen=max(lens)),
        ratio=ratio,
        total_rows=total,
    ).segments
    q = torch.randn(1, total, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    index_k = torch.randn(1, segments[-1].key_end, DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(1, total, HEADS, device="cuda") * HEADS**-0.5 * DIM**-0.5
    widest = max(s.key_end - s.key_start for s in segments)
    flags = torch.rand(1, total, -(-widest // BLOCK), device="cuda") < 0.3 if candidates == "user" else None
    options = _options(topk_backend, candidates)

    packed_idx, packed_flags = indexer_select_packed(q, index_k, weights, segments, flags, ratio=ratio, **options)

    for segment in segments:
        rows, keys = slice(segment.row_start, segment.row_end), slice(segment.key_start, segment.key_end)
        n_keys, num_blocks = segment.key_end - segment.key_start, -(-(segment.key_end - segment.key_start) // BLOCK)
        if n_keys == 0:
            assert bool((packed_idx[:, rows] == -1).all())
            continue
        lens_alone = torch.arange(1, segment.row_end - segment.row_start + 1, device="cuda") // ratio
        alone_idx, alone_flags = indexer_select(
            q[:, rows],
            index_k[:, keys],
            weights[:, rows],
            lens_alone,
            flags[:, rows, :num_blocks] if flags is not None else None,
            **options,
        )
        assert torch.equal(packed_idx[:, rows], torch.where(alone_idx >= 0, alone_idx + segment.key_start, -1))
        if candidates == "source":
            assert torch.equal(packed_flags[:, rows, :num_blocks], alone_flags)
            assert not bool(packed_flags[:, rows, num_blocks:].any())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
