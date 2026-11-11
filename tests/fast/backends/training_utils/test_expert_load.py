from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-b-cpu", labels=["replay"])

import pytest
import torch

from miles.backends.training_utils.replay.expert_load import ExpertParallelLoad, expert_parallel_load


def test_the_busiest_rank_is_the_one_holding_most_of_a_layers_copies():
    routed_experts = torch.tensor(
        [
            [[0, 1], [2, 4]],
            [[1, 0], [6, 3]],
            [[2, 7], [0, 5]],
        ],
        dtype=torch.int32,
    )

    load = expert_parallel_load(routed_experts, num_experts=8, ep_size=4)

    assert load == ExpertParallelLoad(
        num_tokens=3, layer=0, ep_rank=0, copies=4, share=pytest.approx(4 / 6), tokens_without_routing=0
    )


def test_a_token_without_routing_is_counted_and_sends_no_copies():
    routed_experts = torch.tensor([[[-1, -1]], [[4, 5]]], dtype=torch.int32)

    load = expert_parallel_load(routed_experts, num_experts=8, ep_size=4)

    assert load == ExpertParallelLoad(num_tokens=2, layer=0, ep_rank=2, copies=2, share=1.0, tokens_without_routing=1)
