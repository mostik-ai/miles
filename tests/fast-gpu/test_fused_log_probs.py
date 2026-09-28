from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, suite="stage-b-2-gpu-h200", labels=["megatron"], hardware=["hopper", "blackwell"])

"""GPU tests for the Triton kernels behind ``--log-probs-backend fused``.

The kernels must give log_softmax's log-prob, entropy and logits gradient; vocab shards over a
real NCCL group must combine to the full-vocab result; the policy loss must match the torch
backend; and at a long-context shape the op must not hold any vocab-sized buffer of its own.
"""

import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from miles.backends.training_utils.loss_hub import fused_log_probs_triton as kernels
from miles.backends.training_utils.loss_hub.fused_log_probs import fused_log_probs_and_entropy
from miles.backends.training_utils.loss_hub.fused_log_probs_triton import LaunchConfig
from miles.backends.training_utils.loss_hub.math_utils import calculate_log_probs_and_entropy

_WORLD_SIZE = 2


def _reference(logits, rows, targets, temperature):
    log_softmax = torch.log_softmax(logits.index_select(0, rows).float() / temperature, dim=-1)
    log_probs = log_softmax.gather(1, targets.unsqueeze(1)).squeeze(1)
    return log_probs, -(log_softmax.exp() * log_softmax).sum(dim=-1)


def _inputs(n_rows, vocab, dtype, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    logits = (torch.randn(n_rows, vocab, device=device, generator=g) * 3).to(dtype)
    rows = torch.cat([torch.arange(3, n_rows // 2), torch.arange(n_rows // 2 + 5, n_rows)]).to(device)
    targets = torch.randint(0, vocab, (rows.numel(),), device=device, generator=g)
    return logits, rows, targets


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("vocab", [50_001, 129_280])
@pytest.mark.parametrize("temperature", [1.0, 0.7])
@pytest.mark.parametrize("entropy_requires_grad", [True, False])
@pytest.mark.parametrize("inplace_backward", [True, False])
def test_kernels_match_log_softmax(dtype, vocab, temperature, entropy_requires_grad, inplace_backward):
    logits, rows, targets = _inputs(300, vocab, dtype, "cuda")
    gen = torch.Generator(device="cuda").manual_seed(1)
    g = torch.randn(rows.numel(), device="cuda", generator=gen)
    c = torch.randn(rows.numel(), device="cuda", generator=gen)

    ref_leaf = logits.clone().requires_grad_(True)
    ref_log_probs, ref_entropy = _reference(ref_leaf, rows, targets, temperature)
    ((ref_log_probs * g).sum() + (ref_entropy * c).sum() * entropy_requires_grad).backward()

    leaf = logits.clone().requires_grad_(True)
    log_probs, entropy = fused_log_probs_and_entropy(
        leaf * 1,
        rows,
        targets,
        tp_group=None,
        temperature=temperature,
        with_entropy=True,
        entropy_requires_grad=entropy_requires_grad,
        inplace_backward=inplace_backward,
    )
    ((log_probs * g).sum() + ((entropy * c).sum() if entropy_requires_grad else 0)).backward()

    torch.testing.assert_close(log_probs, ref_log_probs.detach(), rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(entropy, ref_entropy.detach(), rtol=1e-5, atol=5e-5)
    grad_tol = dict(rtol=1e-5, atol=1e-5) if dtype == torch.float32 else dict(rtol=1e-2, atol=1e-3)
    torch.testing.assert_close(leaf.grad.float(), ref_leaf.grad.float(), **grad_tol)
    unscored = torch.ones(logits.size(0), dtype=torch.bool, device="cuda")
    unscored[rows] = False
    assert (leaf.grad[unscored] == 0).all()


def test_forward_only_matches_log_softmax():
    logits, rows, targets = _inputs(257, 129_280, torch.bfloat16, "cuda")
    with torch.no_grad():
        log_probs, entropy = fused_log_probs_and_entropy(
            logits, rows, targets, tp_group=None, temperature=0.6, with_entropy=True
        )
    ref_log_probs, ref_entropy = _reference(logits, rows, targets, 0.6)
    torch.testing.assert_close(log_probs, ref_log_probs, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(entropy, ref_entropy, rtol=1e-5, atol=5e-5)


@pytest.mark.parametrize(
    "launch", [LaunchConfig(1024, 1), LaunchConfig(2048, 2), LaunchConfig(4096, 8), LaunchConfig(8192, 16)], ids=str
)
def test_every_launch_shape_matches_log_softmax(launch):
    """Each GPU family may run its own launch shape; every shape must give log_softmax's values and
    gradient, and write every element of the gradient."""
    logits, rows, targets = _inputs(64, 50_001, torch.bfloat16, "cuda", seed=15)
    ref_log_probs, ref_entropy = _reference(logits, rows, targets, 0.7)
    row_max, row_sum, row_dsum, target = kernels.row_statistics(
        logits, rows, targets, vocab_start=0, temperature=0.7, with_entropy=True, launch=launch
    )
    log_sum, mean = torch.log(row_sum), row_dsum / row_sum
    torch.testing.assert_close(target - log_sum, ref_log_probs, rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(log_sum - mean, ref_entropy, rtol=1e-5, atol=5e-5)

    gen = torch.Generator(device="cuda").manual_seed(16)
    g = torch.randn(rows.numel(), device="cuda", generator=gen)
    c = torch.randn(rows.numel(), device="cuda", generator=gen)
    ref_leaf = logits.clone().requires_grad_(True)
    ref_lp, ref_ent = _reference(ref_leaf, rows, targets, 0.7)
    ((ref_lp * g).sum() + (ref_ent * c).sum()).backward()
    grad = torch.full_like(logits, float("nan"))  # any element the kernels miss stays NaN
    one_minus_p = -torch.expm1(target - log_sum)
    kernels.write_logits_grad(
        grad,
        logits,
        rows,
        targets,
        row_max,
        log_sum,
        mean,
        one_minus_p,
        g,
        c,
        vocab_start=0,
        temperature=0.7,
        launch=launch,
    )
    kernels.zero_unscored_rows(grad, rows, launch=launch)
    torch.testing.assert_close(grad.float(), ref_leaf.grad.float(), rtol=1e-2, atol=1e-3)


@pytest.mark.parametrize("no_grad_with_entropy", [False, True])
def test_no_grad_log_probs_match_the_training_forward_bitwise(no_grad_with_entropy):
    """The stored old log-probs come from a no-grad pass that usually skips the entropy, while the
    training forward may compute it (a different kernel specialization): they must still agree."""
    logits, rows, targets = _inputs(257, 129_280, torch.bfloat16, "cuda", seed=9)
    with torch.no_grad():
        old_log_probs, _ = fused_log_probs_and_entropy(
            logits, rows, targets, tp_group=None, temperature=0.7, with_entropy=no_grad_with_entropy
        )
    log_probs, _ = fused_log_probs_and_entropy(
        logits.clone().requires_grad_(True), rows, targets, tp_group=None, temperature=0.7, with_entropy=True
    )
    assert torch.equal(old_log_probs, log_probs.detach())


def test_confident_rows_are_as_accurate_as_torch():
    """With the target 10 above every other logit, p_y is close to 1 and 1 - p_y cancels. Against a
    float64 reference, the op must be no less accurate than torch's own float32 log_softmax."""
    logits, rows, targets = _inputs(64, 129_280, torch.float32, "cuda", seed=11)
    logits[rows, targets] = logits[rows].max(dim=-1).values + 10

    def log_probs_and_grad(x, dtype):
        leaf = x.to(dtype).clone().requires_grad_(True)
        lp = torch.log_softmax(leaf.index_select(0, rows) / 0.7, dim=-1).gather(1, targets.unsqueeze(1)).squeeze(1)
        lp.sum().backward()
        return lp.detach().double(), leaf.grad.double()

    exact_lp, exact_grad = log_probs_and_grad(logits, torch.float64)
    torch_lp, torch_grad = log_probs_and_grad(logits, torch.float32)
    leaf = logits.clone().requires_grad_(True)
    fused_lp, _ = fused_log_probs_and_entropy(leaf * 1, rows, targets, tp_group=None, temperature=0.7)
    fused_lp.sum().backward()

    def error(x, exact):
        return (x.double() - exact).abs().max().item()

    assert error(fused_lp.detach(), exact_lp) <= 2 * error(torch_lp, exact_lp) + 1e-7
    assert error(leaf.grad, exact_grad) <= 2 * error(torch_grad, exact_grad) + 1e-7


def test_rows_beyond_int32_offsets():
    """Rows whose element offset passes 2**31 must still be read and written at the right place."""
    vocab = 129_280
    n_rows = 2**31 // vocab + 8  # 2.15e9 elements, 4.3 GB of bf16
    logits = torch.zeros(n_rows, vocab, device="cuda", dtype=torch.bfloat16)
    rows = torch.arange(n_rows - 4, n_rows, device="cuda")
    gen = torch.Generator(device="cuda").manual_seed(13)
    logits[rows] = (torch.randn(rows.numel(), vocab, device="cuda", generator=gen) * 3).to(torch.bfloat16)
    targets = torch.randint(0, vocab, (rows.numel(),), device="cuda", generator=gen)

    ref_leaf = logits[rows].clone().requires_grad_(True)
    ref_log_probs, ref_entropy = _reference(ref_leaf, torch.arange(rows.numel(), device="cuda"), targets, 0.8)
    (ref_log_probs.sum() + ref_entropy.sum()).backward()

    leaf = logits.requires_grad_(True)
    log_probs, entropy = fused_log_probs_and_entropy(
        leaf, rows, targets, tp_group=None, temperature=0.8, with_entropy=True
    )
    (log_probs.sum() + entropy.sum()).backward()

    torch.testing.assert_close(log_probs, ref_log_probs.detach(), rtol=1e-5, atol=2e-5)
    torch.testing.assert_close(entropy, ref_entropy.detach(), rtol=1e-5, atol=5e-5)
    torch.testing.assert_close(leaf.grad[rows].float(), ref_leaf.grad.float(), rtol=1e-2, atol=1e-3)
    assert leaf.grad[: n_rows - rows.numel()].abs().max().item() == 0


def test_inplace_backward_invalidates_the_logits():
    """The Triton kernels write the gradient into the logits; the version bump makes any second
    reader of those logits fail instead of reading the gradient."""
    logits, rows, targets = _inputs(64, 50_001, torch.bfloat16, "cuda")
    leaf = logits.clone().requires_grad_(True)
    log_probs, _ = fused_log_probs_and_entropy(leaf * 1, rows, targets, tp_group=None, inplace_backward=True)
    log_probs.sum().backward(retain_graph=True)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        log_probs.sum().backward()


def test_policy_gradient_matches_the_torch_backend(tmp_path):
    """The torch backend rounds its logits gradient to bf16 (Megatron's fused CE), so compare at bf16."""
    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        logits, rows, targets = _inputs(513, 129_280, torch.bfloat16, "cuda", seed=3)
        gen = torch.Generator(device="cuda").manual_seed(4)
        g = torch.randn(rows.numel(), device="cuda", generator=gen)
        c = torch.randn(rows.numel(), device="cuda", generator=gen)
        grads = {}
        for backend in ("torch", "fused"):
            leaf = logits.clone().requires_grad_(True)
            if backend == "fused":
                lp, ent = fused_log_probs_and_entropy(
                    leaf * 1, rows, targets, tp_group=None, temperature=0.8, with_entropy=True, inplace_backward=True
                )
            else:
                lp, ent = calculate_log_probs_and_entropy(
                    (leaf * 1).index_select(0, rows), targets, dist.group.WORLD, with_entropy=True, temperature=0.8
                )
                lp = lp.squeeze(-1)
            ((lp * g).sum() + (ent * c).sum()).backward()
            grads[backend] = (lp.detach(), ent.detach(), leaf.grad.float())
        (t_lp, t_ent, t_grad), (f_lp, f_ent, f_grad) = grads["torch"], grads["fused"]
        torch.testing.assert_close(f_lp, t_lp, rtol=1e-5, atol=2e-5)
        torch.testing.assert_close(f_ent, t_ent, rtol=1e-5, atol=5e-5)
        # the torch backend rounds the log-prob and the entropy gradient to bf16 separately, then
        # adds them, so it sits up to about two bf16 ulps from the exact gradient
        torch.testing.assert_close(f_grad, t_grad, rtol=2e-2, atol=5e-3)
    finally:
        dist.destroy_process_group()


def test_the_op_holds_no_vocab_sized_buffer(tmp_path):
    """At 8192 rows x 129280 vocab (2 GiB of bf16 logits), the fused op adds under 1% of the logits
    to the peak, while the torch path's fp32 copies and saved softmax add several times the logits."""
    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        n_rows, vocab = 8192, 129_280
        logits = torch.randn(n_rows, vocab, device="cuda").to(torch.bfloat16)
        rows = torch.arange(n_rows, device="cuda")
        targets = torch.randint(0, vocab, (n_rows,), device="cuda")
        logits_bytes = logits.numel() * logits.element_size()
        extra = {}
        for backend in ("torch", "fused"):
            model_logits = logits.clone().requires_grad_(True)  # stands for the model output
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = torch.cuda.memory_allocated()
            if backend == "fused":
                lp, ent = fused_log_probs_and_entropy(
                    model_logits, rows, targets, tp_group=None, with_entropy=True, inplace_backward=True
                )
            else:
                lp, ent = calculate_log_probs_and_entropy(model_logits, targets, dist.group.WORLD, with_entropy=True)
            (lp.sum() + ent.sum()).backward(inputs=[model_logits])
            torch.cuda.synchronize()
            # the leaf's own .grad is one logits-sized buffer in both backends
            extra[backend] = torch.cuda.max_memory_allocated() - before - logits_bytes
            del model_logits, lp, ent
        print(f"extra peak beyond logits and their grad: {extra}", flush=True)
        assert extra["fused"] < 0.01 * logits_bytes
        assert extra["torch"] > 2 * logits_bytes
    finally:
        dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("localhost", 0))
        return sock.getsockname()[1]


def _tp_worker(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        vocab = 129_280
        logits, rows, targets = _inputs(300, vocab, torch.bfloat16, "cuda", seed=5)
        g = torch.randn(rows.numel(), device="cuda", generator=torch.Generator(device="cuda").manual_seed(6))
        c = torch.randn(rows.numel(), device="cuda", generator=torch.Generator(device="cuda").manual_seed(7))
        full = logits.clone().requires_grad_(True)
        ref_log_probs, ref_entropy = _reference(full, rows, targets, 0.9)
        ((ref_log_probs * g).sum() + (ref_entropy * c).sum()).backward()

        width = vocab // world_size
        shard = logits[:, rank * width : (rank + 1) * width].clone().requires_grad_(True)
        log_probs, entropy = fused_log_probs_and_entropy(
            shard * 1,
            rows,
            targets,
            tp_group=dist.group.WORLD,
            temperature=0.9,
            with_entropy=True,
            inplace_backward=True,
        )
        ((log_probs * g).sum() + (entropy * c).sum()).backward()

        torch.testing.assert_close(log_probs, ref_log_probs.detach(), rtol=1e-5, atol=2e-5)
        torch.testing.assert_close(entropy, ref_entropy.detach(), rtol=1e-5, atol=5e-5)
        torch.testing.assert_close(
            shard.grad.float(), full.grad[:, rank * width : (rank + 1) * width].float(), rtol=1e-2, atol=1e-3
        )
    finally:
        dist.destroy_process_group()


def test_vocab_shards_over_nccl_combine_like_the_full_vocab():
    if torch.cuda.device_count() < _WORLD_SIZE:
        raise RuntimeError(f"requires {_WORLD_SIZE} GPUs, found {torch.cuda.device_count()}")
    mp.spawn(_tp_worker, args=(_WORLD_SIZE, _free_port()), nprocs=_WORLD_SIZE, join=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
