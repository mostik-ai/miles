from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from tests.ci.ci_register import register_cpu_ci

from miles.backends.training_utils.loss.objective import loss_function

from .loss_test_utils import deep_clone, make_args, make_batch, make_inputs, make_parallel_state

register_cpu_ci(est_time=20, suite="stage-b-cpu", labels=["precision"])

PROMPTS, RESPONSES = [20, 64, 40], [10, 48, 32]
SKIPPED = 1


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    if dist.is_initialized():
        yield
        return
    rendezvous = tmp_path_factory.mktemp("zero-advantage-skip") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _loss_and_gradient(args, batch, logits):
    logits = logits.clone().requires_grad_(True)
    loss, _, _ = loss_function(args, batch, 1, logits, apply_megatron_loss_scaling=True, num_rollouts=len(PROMPTS))
    loss.backward()
    return loss.detach(), logits.grad


def _per_token_loss_and_gradient(args, batch, logits):
    logits = logits.clone().requires_grad_(True)
    loss, num_tokens, _ = loss_function(args, batch, 1, logits, apply_megatron_loss_scaling=True)
    loss.backward()
    return loss.detach() / num_tokens, logits.grad / num_tokens


def _without_sample(batch: dict, index: int) -> dict:
    return {key: [value[i] for i in range(len(value)) if i != index] for key, value in batch.items()}


@pytest.mark.parametrize("use_tis", [False, True])
def test_leaving_out_a_zero_advantage_sample_leaves_the_update_exactly_unchanged(process_group, use_tis):
    make_parallel_state()
    args = make_args(
        entropy_coef=0.0,
        kl_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        skip_actor_forward_only=True,
        use_tis=use_tis,
    )
    inputs = make_inputs(
        seed=1234, batch_size=3, prompt_lens=PROMPTS, response_lens=RESPONSES, vocab_size=128, args=args
    )
    batch = make_batch(inputs, "policy_loss")
    del batch["log_probs"]
    batch["advantages"] = [
        torch.zeros_like(advantage) if i == SKIPPED else advantage for i, advantage in enumerate(batch["advantages"])
    ]
    logits = deep_clone(inputs["policy_logits"])
    start = sum(inputs["total_lens"][:SKIPPED])
    stop = start + inputs["total_lens"][SKIPPED]

    whole_loss, whole_gradient = _loss_and_gradient(args, batch, logits)
    kept_loss, kept_gradient = _loss_and_gradient(
        args, _without_sample(batch, SKIPPED), torch.cat([logits[:, :start], logits[:, stop:]], dim=1)
    )

    assert torch.equal(whole_loss, kept_loss)
    assert torch.count_nonzero(whole_gradient[:, start:stop]) == 0
    assert torch.equal(torch.cat([whole_gradient[:, :start], whole_gradient[:, stop:]], dim=1), kept_gradient)
    assert torch.count_nonzero(kept_gradient) > 0


def test_leaving_out_a_zero_advantage_sample_leaves_the_per_token_update_unchanged(process_group):
    make_parallel_state()
    args = make_args(
        entropy_coef=0.0,
        kl_coef=0.0,
        observe_training_entropy=False,
        true_on_policy_mode=False,
        skip_actor_forward_only=True,
        calculate_per_token_loss=True,
    )
    inputs = make_inputs(
        seed=1234, batch_size=3, prompt_lens=PROMPTS, response_lens=RESPONSES, vocab_size=128, args=args
    )
    batch = make_batch(inputs, "policy_loss")
    del batch["log_probs"]
    batch["advantages"] = [
        torch.zeros_like(advantage) if i == SKIPPED else advantage for i, advantage in enumerate(batch["advantages"])
    ]
    logits = deep_clone(inputs["policy_logits"])
    start = sum(inputs["total_lens"][:SKIPPED])
    stop = start + inputs["total_lens"][SKIPPED]
    kept_batch = _without_sample(batch, SKIPPED)
    kept_batch["computed_loss_token_share"] = (sum(RESPONSES) - RESPONSES[SKIPPED]) / sum(RESPONSES)

    whole_loss, whole_gradient = _per_token_loss_and_gradient(args, batch, logits)
    kept_loss, kept_gradient = _per_token_loss_and_gradient(
        args, kept_batch, torch.cat([logits[:, :start], logits[:, stop:]], dim=1)
    )

    torch.testing.assert_close(whole_loss, kept_loss)
    # the backward pass rounds in half precision, so a loss scaled before it differs in the last bits
    torch.testing.assert_close(
        torch.cat([whole_gradient[:, :start], whole_gradient[:, stop:]], dim=1), kept_gradient, rtol=1e-2, atol=2e-4
    )
    assert torch.count_nonzero(kept_gradient) > 0
