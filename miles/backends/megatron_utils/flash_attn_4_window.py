"""Transformer Engine passes FlashAttention 4 its FlashAttention 2 window, where -1 means unbounded, so every causal
mask arrives as (-1, 0):
https://github.com/NVIDIA/TransformerEngine/blob/release_v2.17/transformer_engine/pytorch/attention/dot_product_attention/backends.py
FlashAttention 4 marks an unbounded side with None and reads (-1, 0) as a local window that ends before the diagonal, so
every output row is zero and the SM100 head_dim 256 backward refuses local attention:
https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py
"""

import functools

FLASH_ATTN_4_FUNCTIONS = ("flash_attn_func_v4", "flash_attn_varlen_func_v4")


def unbounded_as_none(window_size: tuple[int | None, int | None] | None) -> tuple[int | None, int | None] | None:
    if window_size is None:
        return None
    left, right = window_size
    return (None if left is not None and left < 0 else left, None if right is not None and right < 0 else right)


def patch_flash_attn_4_window() -> None:
    from transformer_engine.pytorch.attention.dot_product_attention import backends

    for name in FLASH_ATTN_4_FUNCTIONS:
        function = getattr(backends, name, None)
        if function is None or getattr(function, "miles_unbounded_as_none", False):
            continue

        @functools.wraps(function)
        def call(*args, window_size=(None, None), _function=function, **kwargs):
            return _function(*args, window_size=unbounded_as_none(window_size), **kwargs)

        call.miles_unbounded_as_none = True
        setattr(backends, name, call)
