"""CPU tests for ``--log-probs-backend fused``.

The op's torch path shares the autograd function and the tensor-parallel combine with the Triton
path, so these check the math, the gradient, the vocab-shard combine over a real gloo group, and
that the losses and the all-gather-CP layout give the torch backend's results. The Triton kernels
themselves are checked in tests/fast-gpu/test_fused_log_probs.py.
"""

from functools import partial

import pytest
import torch
import torch.distributed as dist
from tests.fast.dist_utils import init_gloo, run_multiprocess

from miles.backends.training_utils.cp_utils import all_gather_with_cp
from miles.backends.training_utils.loss import loss_function
from miles.backends.training_utils.loss_hub.fused_log_probs import fused_log_probs_and_entropy
from miles.backends.training_utils.loss_hub.logit_processors import get_log_probs_and_entropy
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

from .loss_test_utils import make_args, make_batch, make_inputs, make_parallel_state

VOCAB_SIZE = 64
# Megatron's fused CE (the torch backend) rounds its logits gradient to bf16 even for fp32
# logits (fused_cross_entropy.py, calculate_gradients), so gradients are compared with it at
# bf16 precision; the exact comparison is against log_softmax.
_TORCH_BACKEND_GRAD_TOL = dict(rtol=2e-2, atol=2e-3)


def _reference(logits, rows, targets, temperature):
    """Log-prob and entropy of ``logits / temperature`` at ``rows``, straight from log_softmax."""
    log_softmax = torch.log_softmax(logits.index_select(0, rows).float() / temperature, dim=-1)
    log_probs = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
    return log_probs, -(log_softmax.exp() * log_softmax).sum(dim=-1)


def _logits_and_rows(n_rows=12, vocab=VOCAB_SIZE, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = (torch.randn(n_rows, vocab, generator=g) * 4).to(dtype)
    rows = torch.tensor([1, 2, 3, 7, 8, 11])
    targets = torch.randint(0, vocab, (rows.numel(),), generator=g)
    return logits, rows, targets


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("entropy_requires_grad", [True, False])
@pytest.mark.parametrize("inplace_backward", [True, False])
def test_op_matches_log_softmax_and_its_gradient(dtype, temperature, entropy_requires_grad, inplace_backward):
    logits, rows, targets = _logits_and_rows(dtype=dtype)
    g = torch.randn(rows.numel())
    c = torch.randn(rows.numel())

    ref_leaf = logits.clone().requires_grad_(True)
    ref_log_probs, ref_entropy = _reference(ref_leaf, rows, targets, temperature)
    ((ref_log_probs * g).sum() + (ref_entropy * c).sum() * entropy_requires_grad).backward()

    leaf = logits.clone().requires_grad_(True)
    log_probs, entropy = fused_log_probs_and_entropy(
        leaf * 1,  # model logits are a non-leaf; the in-place gradient lands in this buffer
        rows,
        targets,
        tp_group=None,
        temperature=temperature,
        with_entropy=True,
        entropy_requires_grad=entropy_requires_grad,
        inplace_backward=inplace_backward,
    )
    ((log_probs * g).sum() + ((entropy * c).sum() if entropy_requires_grad else 0)).backward()

    torch.testing.assert_close(log_probs, ref_log_probs.detach(), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy, ref_entropy.detach(), rtol=1e-5, atol=1e-5)
    assert entropy.requires_grad == entropy_requires_grad
    tol = 1e-5 if dtype == torch.float32 else 1e-2
    torch.testing.assert_close(leaf.grad.float(), ref_leaf.grad.float(), rtol=tol, atol=tol)
    unscored = torch.ones(logits.size(0), dtype=torch.bool)
    unscored[rows] = False
    assert (leaf.grad[unscored] == 0).all()


def test_no_grad_returns_values_only():
    logits, rows, targets = _logits_and_rows()
    with torch.no_grad():
        log_probs, entropy = fused_log_probs_and_entropy(logits, rows, targets, tp_group=None, with_entropy=True)
    ref_log_probs, ref_entropy = _reference(logits, rows, targets, 1.0)
    torch.testing.assert_close(log_probs, ref_log_probs, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-5, atol=1e-5)
    assert log_probs.grad_fn is None


def test_no_rows_still_gives_the_logits_a_gradient():
    """A rank that scores nothing must still backprop through the model, or CP collectives hang."""
    leaf = torch.randn(5, VOCAB_SIZE, requires_grad=True)
    empty = torch.empty(0, dtype=torch.long)
    log_probs, _ = fused_log_probs_and_entropy(leaf * 1, empty, empty, tp_group=None, inplace_backward=True)
    log_probs.sum().backward()
    assert leaf.grad is not None and (leaf.grad == 0).all()


def _shard_worker(rank, world_size, port, temperature):
    init_gloo(rank, world_size, port=port)
    try:
        logits, rows, targets = _logits_and_rows(vocab=VOCAB_SIZE)
        g = torch.randn(rows.numel(), generator=torch.Generator().manual_seed(1))
        c = torch.randn(rows.numel(), generator=torch.Generator().manual_seed(2))
        full = logits.clone().requires_grad_(True)
        ref_log_probs, ref_entropy = _reference(full, rows, targets, temperature)
        ((ref_log_probs * g).sum() + (ref_entropy * c).sum()).backward()

        shard_width = VOCAB_SIZE // world_size
        shard = logits[:, rank * shard_width : (rank + 1) * shard_width].clone().requires_grad_(True)
        log_probs, entropy = fused_log_probs_and_entropy(
            shard * 1,
            rows,
            targets,
            tp_group=dist.group.WORLD,
            temperature=temperature,
            with_entropy=True,
            inplace_backward=True,
        )
        ((log_probs * g).sum() + (entropy * c).sum()).backward()

        torch.testing.assert_close(log_probs, ref_log_probs.detach(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(entropy, ref_entropy.detach(), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            shard.grad, full.grad[:, rank * shard_width : (rank + 1) * shard_width], rtol=1e-5, atol=1e-5
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("temperature", [1.0, 0.6])
def test_vocab_shards_combine_like_the_full_vocab(temperature):
    run_multiprocess(partial(_shard_worker, temperature=temperature), world_size=2)


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    """The torch backend's fused CE reaches a collective; give it a 1-rank group."""
    if dist.is_initialized():
        yield
        return
    rendezvous = tmp_path_factory.mktemp("fused-log-probs") / "process-group"
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("loss_type", ["policy_loss", "sft_loss"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_losses_match_the_torch_backend(process_group, loss_type, dtype):
    make_parallel_state()
    prompt_lens, response_lens = [5, 9, 3], [7, 4, 6]
    results = {}
    for backend in ("torch", "fused"):
        args = make_args(
            loss_type=loss_type,
            true_on_policy_mode=False,
            log_probs_backend=backend,
            rollout_temperature=0.8,
            entropy_coef=0.01,
        )
        inputs = make_inputs(7, len(prompt_lens), prompt_lens, response_lens, VOCAB_SIZE, args)
        leaf = inputs["policy_logits"].to(dtype).requires_grad_(True)
        loss, _, log = loss_function(args, make_batch(inputs, loss_type), 1, leaf * 1)
        loss.backward()
        results[backend] = (loss.detach(), dict(zip(log["keys"], log["values"][1:].tolist(), strict=True)), leaf.grad)

    (torch_loss, torch_log, torch_grad), (fused_loss, fused_log, fused_grad) = results["torch"], results["fused"]
    torch.testing.assert_close(fused_loss, torch_loss, rtol=1e-5, atol=1e-5)
    for key, value in torch_log.items():
        assert fused_log[key] == pytest.approx(value, rel=1e-4, abs=1e-5), key
    torch.testing.assert_close(fused_grad.float(), torch_grad.float(), **_TORCH_BACKEND_GRAD_TOL)


# (prompt_lens, response_lens): totals divide by 2 * cp. In the second case rank 0's contiguous
# half holds no response logits, so only the log-prob op keeps its logits in the graph.
CP_CASES = [([16, 2], [8, 6]), ([40], [24])]


def _cp2_worker(rank, world_size, port, prompt_lens, response_lens):
    init_gloo(rank, world_size, port=port)
    try:
        tp_group = [dist.new_group([r]) for r in range(world_size)][rank]
        trivial = GroupInfo(rank=0, size=1, group=None)
        cp = GroupInfo(rank=rank, size=world_size, group=dist.group.WORLD)
        set_parallel_state(
            ParallelState(
                intra_dp=trivial,
                intra_dp_cp=cp,
                cp=cp,
                tp=GroupInfo(rank=0, size=1, group=tp_group),
                pp=trivial,
                ep=trivial,
                etp=trivial,
                indep_dp=trivial,
                is_pp_last_stage=True,
            )
        )
        base = make_args(loss_type="sft_loss", true_on_policy_mode=False, allgather_cp=True, rollout_temperature=0.7)
        inputs = make_inputs(42, len(prompt_lens), prompt_lens, response_lens, VOCAB_SIZE, base)
        t_local = inputs["policy_logits"].size(1) // world_size
        local = inputs["policy_logits"][:, rank * t_local : (rank + 1) * t_local]

        outputs = {}
        for backend in ("torch", "fused"):
            args = make_args(**{**vars(base), "log_probs_backend": backend})
            leaf = local.clone().requires_grad_(True)
            res = get_log_probs_and_entropy(
                leaf * 1,
                args=args,
                unconcat_tokens=inputs["unconcat_tokens"],
                total_lengths=inputs["total_lens"],
                response_lengths=response_lens,
            )
            leaf_for_loss = local.clone().requires_grad_(True)
            loss, _, _ = loss_function(args, make_batch(inputs, "sft_loss"), 1, leaf_for_loss * 1)
            loss.backward()
            full = [
                all_gather_with_cp(lp.detach(), total_len, response_len)
                for lp, total_len, response_len in zip(
                    res["log_probs"], inputs["total_lens"], response_lens, strict=True
                )
            ]
            outputs[backend] = (full, loss.detach(), leaf_for_loss.grad)

        for fused_lp, torch_lp in zip(outputs["fused"][0], outputs["torch"][0], strict=True):
            torch.testing.assert_close(fused_lp, torch_lp, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(outputs["fused"][1], outputs["torch"][1], rtol=1e-5, atol=1e-5)
        # the backward reached this rank's logits without the 0 * logits.sum() anchor
        assert outputs["fused"][2] is not None
        torch.testing.assert_close(outputs["fused"][2], outputs["torch"][2], **_TORCH_BACKEND_GRAD_TOL)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(("prompt_lens", "response_lens"), CP_CASES, ids=["split_responses", "empty_rank"])
def test_allgather_cp2_matches_the_torch_backend(prompt_lens, response_lens):
    run_multiprocess(partial(_cp2_worker, prompt_lens=prompt_lens, response_lens=response_lens), world_size=2)
