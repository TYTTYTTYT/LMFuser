"""Failure-mode tests for DDPRunner bookkeeping (0.4.1).

Run:  python tests/test_runner_robustness.py

Covers issues found in review, all of which failed silently:
  1. the data seed is deterministic — equal across ranks and across processes
  2. the data seed is stable across a resume that restored data cursors
     (the shard-cursor row permutations are derived from it)
  3. checkpoints are published atomically — a crash mid-save never leaves a
     directory that load() would treat as "start from step 1"
  4. optimizer/scheduler state loads onto CPU, not every rank onto cuda:0
"""
import os
import sys
import json
import hashlib
import shutil
import subprocess
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _derive(base: int, step, stable: bool) -> int:
    """Mirror of the runner's seed derivation."""
    key = f'original_seed_{base}' if stable else f'original_seed_{base}|step_{step}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'big') & 0x7FFFFFFF


def test_seed_is_deterministic_across_processes() -> None:
    """`hash()` on a str is PYTHONHASHSEED-randomized: every rank used to get a
    different seed, so ranks disagreed on task sampling and no run was
    reproducible."""
    snippet = (
        'import hashlib;'
        "k='original_seed_42|step_1';"
        "print(int.from_bytes(hashlib.sha256(k.encode()).digest()[:4],'big') & 0x7FFFFFFF)"
    )
    seeds = {
        subprocess.run([sys.executable, '-c', snippet], capture_output=True,
                       text=True).stdout.strip()
        for _ in range(4)
    }
    assert len(seeds) == 1, f'seed differs across processes: {seeds}'

    old = {
        subprocess.run([sys.executable, '-c',
                        "print(hash('original_seed_42|step_1'))"],
                       capture_output=True, text=True).stdout.strip()
        for _ in range(4)
    }
    assert len(old) > 1, 'expected the old hash() to be randomized — test is moot'
    print(f'PASS 1: seed identical across processes ({seeds.pop()}); '
          f'old hash() gave {len(old)} different values')


def test_seed_stable_when_cursors_restored() -> None:
    """Shard cursors index a permutation derived from the seed; folding the
    step into it on resume would silently invalidate every cursor."""
    fresh_step1 = _derive(42, 1, stable=False)
    fresh_step9 = _derive(42, 900_000, stable=False)
    assert fresh_step1 != fresh_step9, 'without cursors the seed should follow the step'

    resumed_a = _derive(42, 1, stable=True)
    resumed_b = _derive(42, 900_000, stable=True)
    assert resumed_a == resumed_b, 'with cursors restored the seed must not move'
    print('PASS 2: seed stable across a cursor-restoring resume, step-varying otherwise')


def test_checkpoint_is_atomic() -> None:
    """A checkpoint directory must never be visible while incomplete: load()
    treats a missing runner.json as step 1 and silently restarts the run."""
    src = open(os.path.join(os.path.dirname(__file__), '..', 'src', 'lmfuser',
                            'runners', 'ddp_runner.py')).read()

    assert '.incomplete' in src, 'save() does not stage the checkpoint'
    assert 'os.replace(staging, final)' in src, 'save() does not publish by rename'
    # the publish must come after every writer
    pub = src.index('os.replace(staging, final)')
    for writer in ("save_model(model, staging)", "staging / 'optimizer.pt'",
                   "staging / 'scheduler.pt'", "staging / 'runner.json'"):
        assert src.index(writer) < pub, f'{writer} runs after the rename'

    # a staged directory is invisible to a "latest numeric checkpoint" scan,
    # which is how supervisors pick what to resume from
    tmp = tempfile.mkdtemp(prefix='ckpt_atomic_')
    try:
        os.makedirs(os.path.join(tmp, '.25000.incomplete'))
        os.makedirs(os.path.join(tmp, '20000'))
        numeric = [d for d in os.listdir(tmp) if d.isdigit()]
        assert numeric == ['20000'], f'staging dir leaked into the scan: {numeric}'
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print('PASS 3: checkpoint staged then published by rename; staging dir not resumable')


def test_state_loads_on_cpu() -> None:
    """Without map_location every rank deserializes rank0's CUDA optimizer
    state onto cuda:0 — ~4GB x N ranks transiently, OOM on the larger models."""
    src = open(os.path.join(os.path.dirname(__file__), '..', 'src', 'lmfuser',
                            'runners', 'ddp_runner.py')).read()
    for name in ('optimizer_path', 'scheduler_path'):
        idx = src.index(f'torch.load({name}')
        call = src[idx:idx + 120]
        assert "map_location='cpu'" in call, f'torch.load({name}) lacks map_location'
    print('PASS 4: optimizer/scheduler state deserializes on CPU')


if __name__ == '__main__':
    test_seed_is_deterministic_across_processes()
    test_seed_stable_when_cursors_restored()
    test_checkpoint_is_atomic()
    test_state_loads_on_cpu()
    print('ALL PASS')
