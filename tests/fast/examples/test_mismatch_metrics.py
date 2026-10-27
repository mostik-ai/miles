from argparse import Namespace

import pytest
import torch

from examples.infra_features.train_infer_mismatch_helper.mis import compute_mis_weights


def test_metrics_only_needs_no_importance_sampling_configuration() -> None:
    train = [torch.tensor([-1.0, -2.0], requires_grad=True)]
    rollout = [torch.tensor([-1.1, -1.9])]
    masks = [torch.ones(2)]
    weights, returned_masks, metrics = compute_mis_weights(Namespace(use_tis=False), train_log_probs=train, rollout_log_probs=rollout, loss_masks=masks)

    assert weights is None
    assert returned_masks is masks
    assert metrics
    assert all(torch.isfinite(value).all() for values in metrics.values() for value in values)
    assert all(not value.requires_grad for values in metrics.values() for value in values)


def test_metrics_only_keeps_original_loss_masks() -> None:
    masks = [torch.tensor([1.0, 0.0, 1.0])]
    original = masks[0].clone()
    weights, returned_masks, _ = compute_mis_weights(
        Namespace(use_tis=False),
        train_log_probs=[torch.tensor([-1.0, -2.0, -3.0])],
        rollout_log_probs=[torch.tensor([-1.0, -1.0, -2.5])],
        loss_masks=masks,
    )

    assert weights is None
    assert returned_masks is masks
    torch.testing.assert_close(returned_masks[0], original)


def test_metrics_only_cp_preserves_loss_and_gradient(monkeypatch: pytest.MonkeyPatch) -> None:
    from miles.backends.training_utils import parallel
    from miles.utils.ft_utils.process_group_utils import GroupInfo

    from examples.infra_features.train_infer_mismatch_helper.mis import compute_mis_weights_with_cp

    group = GroupInfo(rank=0, size=1, group=None)
    state = parallel.ParallelState(**dict.fromkeys(("intra_dp", "intra_dp_cp", "cp", "tp", "pp", "ep", "etp", "indep_dp"), group))
    monkeypatch.setattr(parallel, "_parallel_state", state)
    loss = torch.tensor([0.25, -0.5], requires_grad=True)
    masks = [torch.tensor([1.0, 0.0])]
    checked_loss, checked_masks, metrics = compute_mis_weights_with_cp(
        Namespace(use_tis=False, qkv_format="thd"),
        pg_loss=loss,
        train_log_probs=[torch.tensor([-1.0, -2.0])],
        rollout_log_probs=[torch.tensor([-1.1, -1.9])],
        loss_masks=masks,
        total_lengths=[3],
        response_lengths=[2],
    )
    assert checked_loss is loss
    assert checked_masks is masks
    assert metrics and all(torch.isfinite(value).all() for value in metrics.values())
    checked_loss.sum().backward()
    torch.testing.assert_close(loss.grad, torch.ones_like(loss))
