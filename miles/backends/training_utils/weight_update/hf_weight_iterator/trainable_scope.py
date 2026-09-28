"""Opt-in trainable-parameter scope for the base-weight stream.

``--update-weight-parameter-scope trainable`` keeps only the update units a trainer actually
trains, so a frozen part of the model (a frozen backbone, a frozen tower, a frozen receiver of
a side module) stops crossing the trainer-to-engine boundary on every sync. The default stays
``all``.

The selection is a late filter over assembled update units: every rank still drives the
exporter in lockstep and only drops units afterwards, so no collective can desynchronize.
Trainer-side gathering and conversion of frozen parameters still happen; pushing the selection
into the conversion tasks is the separate, performance-complete step.
"""

from collections.abc import Iterable, Iterator

import torch

PARAMETER_SCOPES = ("all", "trainable")


def local_trainable_global_names(conversion_tasks: Iterable) -> set[str]:
    """Global Megatron names of the parameters this rank owns and trains.

    Ranks that do not own a parameter hold a name-only stand-in task (``param_weight=None``),
    so ``requires_grad`` is readable on the owner only: the result is a per-rank contribution
    that the caller unions across the model-parallel ranks."""
    return {
        task.global_param_name
        for task in conversion_tasks
        if task is not None and task.param_weight is not None and task.param_weight.requires_grad
    }


def trainable_source_names(conversion_tasks: Iterable, trainable_global_names: set[str]) -> set[str]:
    """The exporter-reported source names of the trainable parameters, for *this* rank's tasks.

    The exporter names a weight by its task's ``param_name``: the local name on the rank owning
    the parameter, the global name on the stand-in tasks of every other rank. Both are mapped
    through the same task list here, so each rank selects the same set of exported weights from
    the one agreed set of global names.

    A layout where one exported name covers both a trainable and a frozen parameter (a local
    name colliding with another stage's global name) cannot be resolved this way and raises."""
    trainable_by_name: dict[str, set[bool]] = {}
    for task in conversion_tasks:
        if task is None:
            continue
        trainable_by_name.setdefault(task.param_name, set()).add(task.global_param_name in trainable_global_names)
    ambiguous = sorted(name for name, flags in trainable_by_name.items() if len(flags) > 1)
    if ambiguous:
        raise RuntimeError(
            "--update-weight-parameter-scope trainable cannot resolve this parallel layout: the "
            f"exported source names {ambiguous[:5]} name both a trainable and a frozen parameter."
        )
    return {name for name, flags in trainable_by_name.items() if True in flags}


def select_trainable_units(
    hf_param_units: Iterable[list[tuple[str, torch.Tensor]]],
    trainable_hf_names: set[str],
) -> Iterator[list[tuple[str, torch.Tensor]]]:
    """Keep the assembled units that carry at least one tensor from a trainable parameter.

    Whole units only: a unit and an atomic update group are indivisible, so a frozen companion
    of a trainable tensor is sent rather than a partial packed or quantized object. Units with
    no trainable source — including checkpoint passthrough tensors, which have no trainer-side
    parameter at all — are dropped. Selecting nothing is a configuration error, not a no-op."""
    selected = 0
    for unit in hf_param_units:
        if any(name in trainable_hf_names for name, _tensor in unit):
            selected += 1
            yield unit
    if not selected:
        raise RuntimeError(
            "--update-weight-parameter-scope trainable selected no weights: no exported update "
            "unit came from a parameter with requires_grad. Check that the model is not fully frozen."
        )
