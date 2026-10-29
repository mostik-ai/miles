import pytest
import torch

pytest.importorskip("megatron.core")


def test_a_warmup_step_logs_the_rate_it_updates_with_not_the_next_steps():
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    from miles.backends.megatron_utils.model import step_learning_rates

    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=0.0)
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=2e-4,
        min_lr=0.0,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="constant",
        start_wd=0.01,
        end_wd=0.01,
        wd_incr_steps=1000,
        wd_incr_style="constant",
    )

    logged = []
    for _ in range(3):
        logged.append(step_learning_rates(optimizer)[0])
        scheduler.step(increment=1)

    assert logged == pytest.approx([0.0, 2e-6, 4e-6])
