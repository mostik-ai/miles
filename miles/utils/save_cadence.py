"""When a checkpoint is due, and whether the publisher admits it now (coral's save cadence).

`--save-interval-minutes` makes a save due on the clock as well as on `--save-interval` rollouts, so the recovery
point of a run is bounded in time rather than in updates. `--save-admission-path` names the publisher's grant: a due
save is taken only while that file exists and consumes it by renaming it to `<path>.<rollout_id>`; without the grant
the save is deferred and retried at every later boundary, never dropped. The external sentinel and the last rollout
save regardless, so a run always ends with a checkpoint.

The two together are the run's cadence: the clock says how often a save may start and the grant says whether the
previous one is durable yet, so saves are spaced by `max(interval, the previous publication)` and never overlap.

The elapsed clock and the deferral live here rather than in the two train loops, which only ask: one trainer process
runs one loop, and both `train.py` and `train_async.py` ask the same question at the same point.
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

# Read by coral's in-attempt watchdog to void the save deadline a boundary armed (coral.launcher.node.Monitor).
DEFERRED = "Save deferred at rollout %d: the publisher has not admitted another checkpoint"

_cadence = {"last_save": None, "deferred": False}


def start():
    """Start the clock, once, where the train loop begins.

    The interval runs from the loop's first rollout, not from its first save: a run whose save interval is 60 minutes
    has its first recovery point an hour in, which is what `--save-interval-minutes` promises and what the publisher's
    grant is paced by. Seeding it at the first save boundary instead would silently add one rollout to every attempt.
    """
    _cadence.update(last_save=time.monotonic(), deferred=False)


def claim_save_admission(path, rollout_id):
    """Take the publisher's grant for this rollout, by renaming it to its claim `<path>.<rollout_id>`.

    Without an admission path every save is admitted. The publisher grants one save at a time, only while the
    checkpoints already saved are durable and the previous save is old enough, and counts each claim, so an
    ungranted save never passes unseen.
    """
    if path is None:
        return True
    try:
        os.rename(path, f"{path}.{rollout_id}")
    except FileNotFoundError:
        return False
    return True


def save_due(args, rollout_id, periodic, external):
    """(save now, save synchronously) at this boundary, carrying the elapsed clock and the deferral.

    A save is due on the sentinel, the timed interval, the periodic count, or because an earlier one was deferred.
    The sentinel and the last rollout save regardless; any other due save needs the publisher's admission and is
    taken at the first boundary that has it.
    """
    now = time.monotonic()
    if _cadence["last_save"] is None:  # a loop that never called start(): behave as if it had, here
        _cadence["last_save"] = now
    last = rollout_id == args.num_rollout - 1
    interval = getattr(args, "save_interval_minutes", None)
    timed = interval is not None and (now - _cadence["last_save"] >= interval * 60 or last)
    due = _cadence["deferred"] or external or timed or periodic
    forced = external or last
    path = getattr(args, "save_admission_path", None)
    claimed = due and claim_save_admission(path, rollout_id)  # only a due save takes the grant
    save = due and (claimed or forced)
    if save and not claimed and path is not None:
        # A forced save takes no grant, but the publisher counts what is saved but not durable by the claims it
        # finds, so it has to leave one anyway (coral.backends.miles.admission.admit).
        open(f"{path}.{rollout_id}", "a").close()
    _cadence["deferred"] = due and not save
    if _cadence["deferred"]:
        logger.info(DEFERRED, rollout_id)
    if save:
        _cadence["last_save"] = now  # the interval runs from one save starting to the next, as the publisher's does
    return save, forced
