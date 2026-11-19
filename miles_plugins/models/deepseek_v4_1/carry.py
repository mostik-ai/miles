"""The state a DeepSeek-V4.1 layer passes to later layers, and how it travels inside the layer output.

A V4.1 layer reads more than the residual streams: the mixing weights ``pre`` of the sublayer before it, the token
ids (for Engram), the source layers' latents and index keys, the last index source layer's top-k and the candidate
blocks. Within one forward they live in ``V41Runtime``. Under pipeline parallelism or full activation recompute each
layer appends the part later layers need to its output columns, after the ``num_residual_streams * hidden_size``
stream columns, and the next layer unpacks it, so the state follows any split of the layers.
"""

import torch


class V41Runtime:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pre = None
        self.input_ids = None
        self.latent = {}
        self.index_k = {}
        self.topk = None
        self.candidates = None


def v41_pack_pre(pre: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    hi = pre.to(dtype)
    lo = (pre - hi.float()).to(dtype)
    return torch.cat([hi, lo], dim=-1)


def v41_unpack_pre(packed: torch.Tensor, n: int) -> torch.Tensor:
    return packed[..., :n].float() + packed[..., n:].float()


def v41_kv_source_layer(config, layer_id: int):
    ratio = config.csa_compress_ratios[layer_id]
    if not ratio:
        return None
    return max(
        layer
        for layer in config.v41_kv_source_layer_ids
        if layer <= layer_id and config.csa_compress_ratios[layer] == ratio
    )


_CARRY_PRE, _CARRY_IDS, _CARRY_LATENT, _CARRY_INDEX_K, _CARRY_TOPK, _CARRY_CAND = range(6)
_CARRY_HEADER_INTS = 48


def _bits_to_bf16(t: torch.Tensor) -> torch.Tensor:
    t = t.detach().contiguous().flatten()
    if t.dtype == torch.bool:
        t = t.view(torch.uint8)
    if t.dtype == torch.uint8:
        if t.numel() % 2:
            t = torch.nn.functional.pad(t, (0, 1))
        return t.view(torch.bfloat16)
    if t.dtype != torch.int32:
        t = t.to(torch.int32)
    return t.view(torch.bfloat16)


class V41IdsOnlyPlan:
    input_ids = True
    latent_keys = ()
    index_k_keys = ()
    topk = False
    candidates = False


class V41CarryPlan:
    def __init__(self, config, layer_id: int):
        n_layers = config.num_layers
        ratios = config.csa_compress_ratios
        later = range(layer_id + 1, n_layers)
        self.latent_keys = sorted(
            {v41_kv_source_layer(config, layer) for layer in later if ratios[layer]} & set(range(layer_id + 1))
        )
        self.index_k_keys = sorted(
            {
                v41_kv_source_layer(config, layer)
                for layer in later
                if ratios[layer] and layer in config.v41_index_source_layer_ids
            }
            & set(range(layer_id + 1))
        )
        nxt = layer_id + 1
        self.topk = nxt < n_layers and ratios[nxt] > 0 and nxt not in config.v41_index_source_layer_ids
        cand = config.v41_candidate_source_layer_id
        self.candidates = 0 <= cand <= layer_id and any(cand < layer and ratios[layer] for layer in later)
        self.candidate_block_size = config.v41_candidate_block_size
        self.input_ids = bool(config.v41_engram_layer_ids) and layer_id < max(config.v41_engram_layer_ids)


def v41_pack_carry(hidden: torch.Tensor, rt, plan: V41CarryPlan) -> torch.Tensor:
    s, b, _ = hidden.shape
    dt = hidden.dtype
    meta, parts = [], []

    def add(kind, key, shape, flat):
        meta.append([kind, key, *shape, *([0] * (3 - len(shape)))])
        parts.append(flat if flat.dtype == dt else flat.to(dt))

    if rt.pre is not None:
        add(_CARRY_PRE, 0, rt.pre.shape, v41_pack_pre(rt.pre, dt).flatten())
    if plan.input_ids:
        add(_CARRY_IDS, 0, rt.input_ids.shape, _bits_to_bf16(rt.input_ids))
    for k in plan.latent_keys:
        add(_CARRY_LATENT, k, rt.latent[k].shape, rt.latent[k].flatten())
    for k in plan.index_k_keys:
        add(_CARRY_INDEX_K, k, rt.index_k[k].shape, rt.index_k[k].flatten())
    if plan.topk:
        add(_CARRY_TOPK, 0, rt.topk.shape, _bits_to_bf16(rt.topk))
    if plan.candidates:
        bs = plan.candidate_block_size
        blocks = rt.candidates[..., ::bs]
        add(_CARRY_CAND, bs, (*rt.candidates.shape[:-1], rt.candidates.shape[-1]), _bits_to_bf16(blocks))
    header = torch.zeros(_CARRY_HEADER_INTS, dtype=torch.int32, device=hidden.device)
    header[0] = len(meta)
    flat_meta = [v for m in meta for v in m]
    header[1 : 1 + len(flat_meta)] = torch.tensor(flat_meta, dtype=torch.int32, device=hidden.device)
    parts.insert(0, header.view(torch.bfloat16))
    flat = torch.cat(parts)
    rows = s * b
    pad = (-flat.numel()) % rows
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    return torch.cat([hidden, flat.view(s, b, -1)], dim=-1)


def v41_unpack_carry(hidden: torch.Tensor, rt, base_cols: int) -> torch.Tensor:
    rt.reset()
    if hidden.shape[-1] == base_cols:
        return hidden
    carry = hidden[..., base_cols:].contiguous().flatten()
    hidden = hidden[..., :base_cols].contiguous()
    header = carry[: 2 * _CARRY_HEADER_INTS].detach().view(torch.int32).tolist()
    off = 2 * _CARRY_HEADER_INTS
    for i in range(header[0]):
        kind, key, d0, d1, d2 = header[1 + 5 * i : 6 + 5 * i]
        if kind == _CARRY_PRE:
            n = 2 * d0 * d1 * d2
            rt.pre = v41_unpack_pre(carry[off : off + n].view(d0, d1, 2 * d2), d2)
        elif kind == _CARRY_IDS:
            n = 2 * d0 * d1
            rt.input_ids = carry[off : off + n].detach().view(torch.int32).view(d0, d1).to(torch.int64)
        elif kind == _CARRY_LATENT:
            n = d0 * d1 * d2
            rt.latent[key] = carry[off : off + n].view(d0, d1, d2)
        elif kind == _CARRY_INDEX_K:
            n = d0 * d1 * d2
            rt.index_k[key] = carry[off : off + n].view(d0, d1, d2)
        elif kind == _CARRY_TOPK:
            n = 2 * d0 * d1 * d2
            rt.topk = carry[off : off + n].detach().view(torch.int32).view(d0, d1, d2).to(torch.int64)
        elif kind == _CARRY_CAND:
            bs = key
            nb = -(-d2 // bs)
            n = (d0 * d1 * nb + 1) // 2
            raw = carry[off : off + n].detach().view(torch.uint8)
            blocks = raw[: d0 * d1 * nb].view(d0, d1, nb).bool()
            rt.candidates = blocks.repeat_interleave(bs, dim=-1)[..., :d2]
        else:
            raise ValueError(f"unknown carry section {kind}")
        off += n
    return hidden
