from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import torch

HIDDEN_FROM_HF = "hidden_from_hf"


def mark_hidden_from_hf(module: torch.nn.Module) -> torch.nn.Module:
    """Keep module, which has no Hugging Face weights, out of every HF load and HF export of the model it is in."""
    setattr(module, HIDDEN_FROM_HF, True)
    return module


@contextmanager
def hide_marked_modules(model: Sequence[torch.nn.Module]) -> Iterator[None]:
    """Detach every module marked with mark_hidden_from_hf, at any depth of the model chunks, while the block runs."""
    marked = [
        (parent, name, child)
        for chunk in model
        for parent in list(chunk.modules())
        for name, child in list(parent.named_children())
        if getattr(child, HIDDEN_FROM_HF, False)
    ]
    for parent, name, _ in marked:
        delattr(parent, name)
    try:
        yield
    finally:
        for parent, name, child in marked:
            parent.add_module(name, child)
