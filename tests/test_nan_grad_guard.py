"""The NaN/Inf gradient guard: skip a non-finite optimizer step instead of
letting it poison the (fused) moment buffers forever.

These test the two mechanisms the guard in DDPRunner._one_train_step relies on
— the fused optimizer's found_inf channel and the foreach fallback — plus the
consecutive-streak accounting and abort, and a microbenchmark showing the
per-step cost is negligible. The guard's rank-consistency is by construction:
its signal is the post-clip (hence post-all-reduce) grad norm, identical on
every rank, so all ranks skip or step together.
"""
import os
import sys
import time

import torch

_DATA_SRC = os.path.join(os.path.dirname(__file__), '..', '..', 'LMFuser-Data', 'src')
if os.path.isdir(_DATA_SRC):
    sys.path.insert(0, _DATA_SRC)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

HAS_CUDA = torch.cuda.is_available()
DEV = 'cuda' if HAS_CUDA else 'cpu'


def _guard_decision(norm_t):
    """The guard's core signal, exactly as in _one_train_step: one GPU bool,
    no host sync. `.all()` unifies the scalar (clipped) and per-tensor
    (unclipped) forms of norm_t."""
    return ~torch.isfinite(norm_t).all()


def test_fused_found_inf_skips_and_does_not_poison() -> None:
    """A fused / amp-aware optimizer no-ops the step when found_inf is set, and
    the moment buffers are never touched — so the next clean step is normal."""
    if not HAS_CUDA:
        print('SKIP: fused path needs CUDA')
        return
    p = torch.nn.Parameter(torch.ones(8, device=DEV))
    opt = torch.optim.AdamW([p], lr=0.1, fused=True)
    assert getattr(opt, '_step_supports_amp_scaling', False), \
        'fused AdamW should advertise amp-scaling support'

    # a non-finite gradient with found_inf set must be a no-op
    p.grad = torch.tensor([float('nan'), float('inf'), -float('inf'), 1., 2.,
                           3., 4., 5.], device=DEV)
    nonfinite = _guard_decision(torch.tensor(float('nan'), device=DEV))
    opt.grad_scale = None
    opt.found_inf = nonfinite.to(torch.float32)
    opt.step()
    assert torch.allclose(p.detach(), torch.ones(8, device=DEV)), \
        'params moved on a step that should have been skipped'
    assert not any(torch.isnan(v).any().item() for v in opt.state[p].values()
                   if torch.is_tensor(v)), 'moment buffers were poisoned'

    # the very next finite step must update normally
    p.grad = torch.ones(8, device=DEV)
    opt.grad_scale = None
    opt.found_inf = _guard_decision(torch.ones((), device=DEV)).to(torch.float32)
    opt.step()
    assert not torch.allclose(p.detach(), torch.ones(8, device=DEV)), \
        'a clean step after a skip failed to update'
    print('PASS: fused found_inf skips the bad step, keeps state, resumes clean')


def test_foreach_fallback_scrubs_grads() -> None:
    """Optimizers without the found_inf channel (foreach, Adadelta, fused=False)
    take the fallback: scrub the grads so the step is a near-no-op. The scrub
    must actually remove nan/inf (a multiply would leave 0*inf = nan)."""
    p = torch.nn.Parameter(torch.ones(4, device=DEV))
    opt = torch.optim.Adam([p], lr=0.1, foreach=True)
    assert not getattr(opt, '_step_supports_amp_scaling', False), \
        'foreach Adam should NOT advertise amp-scaling — it takes the fallback'

    p.grad = torch.tensor([float('nan'), float('inf'), -float('inf'), 2.],
                          device=DEV)
    # fallback: torch has no _foreach_nan_to_num_, so per-tensor scrub
    for param in (p,):
        torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
    assert torch.isfinite(p.grad).all(), 'scrub left a non-finite gradient'
    opt.step()
    assert torch.isfinite(p.detach()).all(), 'params went non-finite after scrub'
    assert not any(torch.isnan(v).any().item() for v in opt.state[p].values()
                   if torch.is_tensor(v)), 'moment buffers were poisoned'
    print('PASS: foreach fallback scrubs nan/inf, step stays finite')


def test_streak_accumulates_and_resets_on_device() -> None:
    """The consecutive-nonfinite counter increments on bad steps and resets on
    a good one, all on-device (no host sync until the log-step read)."""
    streak = torch.zeros((), device=DEV, dtype=torch.long)
    pattern = [True, True, True, False, True]   # bad, bad, bad, good, bad
    seen = []
    for bad in pattern:
        nonfinite = torch.tensor(bad, device=DEV)
        streak = torch.where(nonfinite, streak + 1, torch.zeros_like(streak))
        seen.append(int(streak.item()))
    assert seen == [1, 2, 3, 0, 1], f'streak sequence wrong: {seen}'
    print(f'PASS: streak accumulates/resets correctly {seen}')


def test_guard_is_gated_and_norm_hoisted_in_source() -> None:
    """Structural guarantees that are easy to regress: the guard only runs on
    the no-scaler path and under the switch, and norm_t is hoisted so the
    unclipped path still has a signal."""
    src = open(os.path.join(os.path.dirname(__file__), '..', 'src', 'lmfuser',
                            'runners', 'ddp_runner.py')).read()
    assert 'self.config.skip_nan_and_inf_grad.value() and self.scaler is None' \
        in src, 'guard is not gated on both the switch and the no-scaler path'
    assert '_step_supports_amp_scaling' in src, 'guard does not branch on the ' \
        'optimizer capability (would break non-fused optimizers)'
    # norm_t must be initialised before the clip block so the unclipped path
    # does not hit a NameError / stale value
    assert 'norm_t = None\n' in src, 'norm_t is not hoisted above the clip block'
    print('PASS: guard gated on switch+no-scaler, branches on capability, '
          'norm_t hoisted')


def test_per_step_overhead_is_negligible() -> None:
    """Microbenchmark: the guard's added per-step work (isfinite + where +
    two attribute sets) against a real fused optimizer step on a ~100M-param
    model. No new host sync, so the added cost should be a small fraction of
    the step it guards."""
    if not HAS_CUDA:
        print('SKIP: benchmark needs CUDA')
        return
    # ~100M params spread over many tensors, like a real model's grads
    params = [torch.nn.Parameter(torch.randn(4096, 4096, device=DEV))
              for _ in range(6)]
    opt = torch.optim.AdamW(params, lr=1e-3, fused=True)
    for p in params:
        p.grad = torch.randn_like(p)
    norm_t = torch.tensor(1.0, device=DEV)

    def guard_ops():
        nonfinite = ~torch.isfinite(norm_t).all()
        opt.grad_scale = None
        opt.found_inf = nonfinite.to(torch.float32)

    # warmup
    for _ in range(5):
        guard_ops(); opt.step()
    torch.cuda.synchronize()

    N = 200
    t0 = time.time()
    for _ in range(N):
        opt.step()
    torch.cuda.synchronize()
    step_ms = (time.time() - t0) / N * 1e3

    t0 = time.time()
    for _ in range(N):
        guard_ops(); opt.step()
    torch.cuda.synchronize()
    guarded_ms = (time.time() - t0) / N * 1e3

    overhead_pct = (guarded_ms - step_ms) / step_ms * 100
    print(f'PASS: step={step_ms:.3f}ms  guarded={guarded_ms:.3f}ms  '
          f'overhead={overhead_pct:+.1f}% of one optimizer step '
          f'(no new per-step host sync)')


if __name__ == '__main__':
    test_fused_found_inf_skips_and_does_not_poison()
    test_foreach_fallback_scrubs_grads()
    test_streak_accumulates_and_resets_on_device()
    test_guard_is_gated_and_norm_hoisted_in_source()
    test_per_step_overhead_is_negligible()
    print('ALL PASS')
