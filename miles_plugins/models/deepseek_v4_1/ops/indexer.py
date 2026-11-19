import functools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from megatron.core.extensions.transformer_engine import TELinear
from megatron.core.transformer.module import MegatronModule, mark_keep_in_fp32
from megatron.core.transformer.transformer_config import TransformerConfig

from miles.kernels.attention.dsa import indexer_logits_sbhd
from miles.kernels.attention.dsa.topk import get_dsa_topk_fn
from miles.kernels.position.rope import apply_rotary_emb
from miles.kernels.quant.fake_quant import fake_quant_fp4
from miles.utils.replay_base import indexer_replay_manager
from miles_plugins.models.deepseek_v4.ops.norm import RMSNorm
from miles_plugins.models.deepseek_v4_1.ops.packed import PackedSegment


def select_candidate_blocks(
    logits: torch.Tensor, compress_lens: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    """The candidate blocks of each query: ``[..., ceil(width / block_size)]`` bool, one flag per block of keys."""
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks, device=logits.device) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    return torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, top.indices, top.values > -torch.inf)


def expand_candidate_blocks(blocks: torch.Tensor, width: int, block_size: int) -> torch.Tensor:
    """Spread each query's candidate block flags over the ``width`` keys they cover."""
    return blocks.repeat_interleave(block_size, dim=-1)[..., :width]


INDEXER_QUERY_CHUNK = int(os.environ.get("MILES_DSV41_INDEXER_QUERY_CHUNK", "8192"))

try:
    import deep_select
except ImportError:  # the torch path stays for images without it
    deep_select = None


@functools.lru_cache(maxsize=1)
def use_deep_select() -> bool:
    if deep_select is None or os.environ.get("MILES_DSV41_DEEP_SELECT", "1") != "1":
        return False
    # the package carries sm100a and sm103a cubins, so anything older keeps torch.topk
    return torch.cuda.get_device_capability()[0] >= 10


def indexer_select(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    compress_lens: torch.Tensor,
    candidates: torch.Tensor | None,
    *,
    is_candidate_source: bool,
    uses_candidates: bool,
    candidate_topk_blocks: int,
    candidate_block_size: int,
    topk: int,
    topk_fn,
    canonical_topk: bool,
    allow_deep_select: bool,
    query_chunk: int | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    bsz, seqlen = q.shape[:2]
    n_kv = index_k.size(1)
    k_t = index_k.transpose(0, 1).contiguous()
    deep_selecting = allow_deep_select and use_deep_select()
    clean_logits = not canonical_topk or deep_selecting or is_candidate_source
    if deep_selecting:
        # deep_select requires aligned score rows; the padded columns sit past every query's length
        align = deep_select.get_stride_requirement()[0] // 4
        pad_kv = -(-n_kv // align) * align - n_kv
        if pad_kv:
            k_t = torch.nn.functional.pad(k_t, (0, 0, 0, 0, 0, pad_kv))
    chunk = seqlen if not query_chunk or query_chunk >= seqlen else query_chunk
    idx_parts: list[torch.Tensor] = []
    cand_parts: list[torch.Tensor] = []
    for s in range(0, seqlen, chunk):
        e = min(s + chunk, seqlen)
        lens = compress_lens[s:e].to(torch.int32)
        row_starts = torch.zeros(e - s, dtype=torch.int32, device=q.device)
        scores = indexer_logits_sbhd(
            q[:, s:e].transpose(0, 1).contiguous(),
            k_t,
            weights[:, s:e].transpose(0, 1).float().contiguous(),
            row_starts,
            lens,
            clean_logits=clean_logits,
        )
        # with clean_logits, indexer_logits_sbhd wrote -inf outside [0, lens) for every query
        if is_candidate_source:
            cand = select_candidate_blocks(
                scores[..., :n_kv], lens.unsqueeze(-1), candidate_topk_blocks, candidate_block_size
            )
            cand_parts.append(cand)
        elif uses_candidates:
            assert candidates is not None
            in_candidates = expand_candidate_blocks(candidates[:, s:e], n_kv, candidate_block_size)
            scores[..., :n_kv].masked_fill_(~in_candidates, -torch.inf)
        if deep_selecting:
            flat = scores.reshape(bsz * (e - s), scores.size(-1))
            row_end = lens.to(torch.int32).repeat(bsz)
            _, idx = deep_select.topk(
                flat,
                min(topk, n_kv),
                end=row_end,
                indices_type=torch.int64,
                return_value=False,
                idx_oob_fill_value=-1,
            )
            idx = idx.reshape(bsz, e - s, -1)
            if n_kv < topk:
                idx = torch.nn.functional.pad(idx, (0, topk - n_kv), value=-1)
            idx_parts.append(idx.sort(dim=-1).values)
            continue
        if n_kv < topk:
            scores = torch.nn.functional.pad(scores, (0, topk - n_kv), value=-torch.inf)
        idx = topk_fn(scores.reshape(bsz * (e - s), scores.size(-1)), topk, row_starts.repeat(bsz), lens.repeat(bsz))
        idx = idx.reshape(bsz, e - s, topk)
        if canonical_topk:
            idx_parts.append(idx.to(torch.int64))
            continue
        idx = idx.sort(dim=-1).values.to(torch.int64)
        idx_parts.append(torch.where(idx < lens.unsqueeze(-1), idx, -1))
    idx = idx_parts[0] if len(idx_parts) == 1 else torch.cat(idx_parts, dim=1)
    if is_candidate_source:
        candidates = cand_parts[0] if len(cand_parts) == 1 else torch.cat(cand_parts, dim=1)
    return idx, candidates


def indexer_select_packed(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
    segments: tuple[PackedSegment, ...],
    candidates: torch.Tensor | None,
    *,
    ratio: int,
    is_candidate_source: bool,
    uses_candidates: bool,
    candidate_block_size: int,
    topk: int,
    **select_options,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """``indexer_select`` run on each sequence of a packed stream alone, as if the stream held only it.

    The picks are absolute compressed rows; the candidate blocks of a query start at its own sequence's first
    compressed row, so they are as wide as the longest sequence's blocks.
    """
    bsz, total_rows = q.shape[:2]
    idx = torch.full((bsz, total_rows, topk), -1, dtype=torch.int64, device=q.device)
    if is_candidate_source:
        widest = max(segment.key_end - segment.key_start for segment in segments)
        candidates = torch.zeros(
            bsz, total_rows, -(-widest // candidate_block_size), dtype=torch.bool, device=q.device
        )
    for segment in segments:
        n_keys = segment.key_end - segment.key_start
        if n_keys == 0:
            continue
        rows = slice(segment.row_start, segment.row_end)
        num_blocks = -(-n_keys // candidate_block_size)
        compress_lens = torch.arange(1, segment.row_end - segment.row_start + 1, device=q.device) // ratio
        segment_idx, segment_candidates = indexer_select(
            q[:, rows],
            index_k[:, segment.key_start : segment.key_end],
            weights[:, rows],
            compress_lens,
            candidates[:, rows, :num_blocks] if uses_candidates else None,
            is_candidate_source=is_candidate_source,
            uses_candidates=uses_candidates,
            candidate_block_size=candidate_block_size,
            topk=topk,
            **select_options,
        )
        idx[:, rows] = torch.where(segment_idx >= 0, segment_idx + segment.key_start, -1)
        if is_candidate_source:
            candidates[:, rows, :num_blocks] = segment_candidates
    return idx, candidates


class DeepSeekV41Indexer(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        layer_id: int,
        head_dim: int,
        compress_ratio: int,
        owns_k: bool,
        is_candidate_source: bool,
        uses_candidates: bool,
    ):
        super().__init__(config=config)
        self.layer_id = layer_id
        self.compress_ratio = compress_ratio
        self.owns_k = owns_k
        self.is_candidate_source = is_candidate_source
        self.uses_candidates = uses_candidates
        self.candidate_topk_blocks = config.v41_candidate_topk_blocks
        self.candidate_block_size = config.v41_candidate_block_size
        self.index_n_heads = config.dsa_indexer_n_heads
        self.index_head_dim = config.dsa_indexer_head_dim
        self.index_topk = config.dsa_indexer_topk
        self.topk_backend = config.miles_dsa_topk_backend
        self.rope_head_dim = config.qk_pos_emb_head_dim
        self.softmax_scale = self.index_head_dim**-0.5
        q_lora_rank = config.q_lora_rank if config.q_lora_rank is not None else config.hidden_size

        self.linear_wq_b = TELinear(
            q_lora_rank,
            self.index_n_heads * self.index_head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        self.linear_weights_proj = TELinear(
            config.hidden_size,
            self.index_n_heads,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            skip_weight_param_allocation=False,
            parallel_mode="duplicated",
        )
        if owns_k:
            self.linear_wk = nn.Linear(head_dim, self.index_head_dim, bias=False, dtype=torch.bfloat16)
            self.k_norm = RMSNorm(self.index_head_dim, config.layernorm_epsilon)
            mark_keep_in_fp32(self.k_norm.weight)
        indexer_replay_manager.register_to_module(
            self, "indexer_replay", stream_idx=sorted(config.v41_index_source_layer_ids).index(layer_id)
        )

    @torch.no_grad()
    def index_keys(self, latent: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        k = self.k_norm(F.linear(latent, self.linear_wk.weight)).clone()
        apply_rotary_emb(k[..., -self.rope_head_dim :], freqs_cis)
        return fake_quant_fp4(k)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        index_k: torch.Tensor,
        freqs_cis: torch.Tensor,
        candidates: torch.Tensor | None,
        *,
        compress_lens: torch.Tensor | None = None,
        segments: tuple[PackedSegment, ...] | None = None,
    ):
        """
        Args:
            compress_lens: for an unpacked batch, the compressed rows each query may read, from the first.
            segments: for a packed stream, its sequences; exactly one of the two is given.
        Returns:
            The top-k compressed rows of each query, -1 past its picks, and the candidate blocks.
        """
        assert (compress_lens is None) != (segments is None), "pass compress_lens for bshd or segments for thd"
        bsz, seqlen, _ = x.shape
        rd = self.rope_head_dim
        q, _ = self.linear_wq_b(qr)
        q = q.view(bsz, seqlen, self.index_n_heads, self.index_head_dim).clone()
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = fake_quant_fp4(q)
        weights, _ = self.linear_weights_proj(x)
        weights = weights * (self.softmax_scale * self.index_n_heads**-0.5)

        replaying = indexer_replay_manager.enabled and indexer_replay_manager.stage != "fallthrough"
        topk_fn = indexer_replay_manager.get_topk_fn(get_dsa_topk_fn(self.topk_backend), return_probs=False)
        options = dict(
            is_candidate_source=self.is_candidate_source,
            uses_candidates=self.uses_candidates,
            candidate_topk_blocks=self.candidate_topk_blocks,
            candidate_block_size=self.candidate_block_size,
            topk=self.index_topk,
            topk_fn=topk_fn,
            canonical_topk=self.topk_backend == "canonical",
            allow_deep_select=self.topk_backend == "torch" and not replaying,
            query_chunk=INDEXER_QUERY_CHUNK,
        )
        if segments is None:
            return indexer_select(q, index_k, weights, compress_lens, candidates, **options)
        if replaying:
            raise NotImplementedError("DeepSeek-V4.1 indexer replay does not support packed (thd) sequences")
        return indexer_select_packed(q, index_k, weights, segments, candidates, ratio=self.compress_ratio, **options)
