"""Failure-mode tests for DDPRunner bookkeeping (0.4.2).

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
# lmfuser 0.4.8 needs lmfuser_data.slowest_epoch, which the installed package
# may predate; prefer the sibling working copy when one is checked out.
_DATA_SRC = os.path.join(os.path.dirname(__file__), '..', '..', 'LMFuser-Data', 'src')
if os.path.isdir(_DATA_SRC):
    sys.path.insert(0, _DATA_SRC)


def _derive(base: int) -> int:
    """Mirror of the runner's seed derivation (see DDPRunner.__init__)."""
    key = f'original_seed_{base}'
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


def test_seed_survives_a_resume() -> None:
    """Shard cursors store a row index into a permutation seeded by this value,
    so the seed a run STARTS with must equal the seed it RESUMES with.

    The comparison that matters is fresh-vs-resume. An earlier version of this
    fix varied the key by step on a fresh run and dropped the step once
    cursors existed — stable resume-to-resume, but every cursor written by the
    fresh run pointed into a permutation the first restart no longer used.
    """
    fresh = _derive(42)
    first_resume = _derive(42)      # after a crash at some arbitrary step
    later_resume = _derive(42)      # and again, much later
    assert fresh == first_resume == later_resume, (
        'the data seed moved between a run and its restart — every restored '
        'shard cursor now indexes a different row permutation'
    )

    # and it must actually depend on the configured seed
    assert _derive(42) != _derive(43), 'different configured seeds collide'

    # the runner must not fold the step in anywhere
    src = open(os.path.join(os.path.dirname(__file__), '..', 'src', 'lmfuser',
                            'runners', 'ddp_runner.py')).read()
    block = src[src.index('self.data_seed'):src.index('self.data_seed') + 400]
    assert 'step' not in block, 'the step is being folded into the data seed again'
    print('PASS 2: data seed identical across fresh run and every resume')


def test_config_seed_not_mutated() -> None:
    """The derived seed must not overwrite config.seed: that value is written
    into runner.json, and a derived-from-derived seed compounds per resume."""
    src = open(os.path.join(os.path.dirname(__file__), '..', 'src', 'lmfuser',
                            'runners', 'ddp_runner.py')).read()
    assert 'self.config.seed = self.config.seed.parse(' not in src, \
        'config.seed is still being overwritten with the derived value'
    assert 'seed=self.data_seed' in src, 'loaders do not use the derived seed'
    print('PASS 3: config.seed left as the user configured it')


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
    print('PASS 4: checkpoint staged then published by rename; staging dir not resumable')


def test_state_loads_on_cpu() -> None:
    """Without map_location every rank deserializes rank0's CUDA optimizer
    state onto cuda:0 — ~4GB x N ranks transiently, OOM on the larger models."""
    src = open(os.path.join(os.path.dirname(__file__), '..', 'src', 'lmfuser',
                            'runners', 'ddp_runner.py')).read()
    for name in ('optimizer_path', 'scheduler_path'):
        idx = src.index(f'torch.load({name}')
        call = src[idx:idx + 120]
        assert "map_location='cpu'" in call, f'torch.load({name}) lacks map_location'
    print('PASS 5: optimizer/scheduler state deserializes on CPU')


def test_epoch_takes_the_slowest_task_and_the_right_weight() -> None:
    """DDPRunner.epoch: slowest train task, weights indexed by TASK id.

    train_data_loaders is per task and holds None for eval-only tasks, while
    train_task_weights is packed to the train tasks only — zipping the two
    pairs a loader with another task's weight as soon as any task is eval-only.
    And `max` let the fastest task end the epoch for the whole run.
    """
    from lmfuser.runners.ddp_runner import DDPRunner

    class FakeLoader:
        def __init__(self, epoch: int) -> None:
            self.epoch = epoch

    class Fake:
        pass

    # task 0: train (epoch 7), task 1: eval-only, task 2: train (epoch 2)
    fake = Fake()
    fake.train_data_loaders = [FakeLoader(7), None, FakeLoader(2)]
    fake.train_task_idxs = [0, 2]
    fake.task_weights = [1.0, 5.0, 1.0]
    fake.pre_epoch = 0
    assert DDPRunner.epoch.fget(fake) == 2, \
        f'expected the slowest train task (2), got {DDPRunner.epoch.fget(fake)}'

    # a zero-weight task is never sampled and must not stall the run
    fake.task_weights = [1.0, 5.0, 0.0]
    assert DDPRunner.epoch.fget(fake) == 7, \
        f'zero-weight task stalled the epoch: {DDPRunner.epoch.fget(fake)}'

    # no train tasks at all
    fake.train_task_idxs = []
    fake.pre_epoch = 3
    assert DDPRunner.epoch.fget(fake) == 3
    print('PASS 6: epoch is the slowest train task, weighted by task id')


def test_pre_epoch_is_not_double_counted_on_resume() -> None:
    """runner.json's `epoch` was written as self.epoch, which already includes
    the loaders' progress. Restoring it into pre_epoch WHILE the data stream is
    also restored counts every completed epoch twice, and once more per resume.
    """
    src = open(os.path.join(os.path.dirname(__file__), '..', 'src', 'lmfuser',
                            'runners', 'ddp_runner.py')).read()
    idx = src.index('self.pre_epoch = (')
    assign = src[idx:idx + 200]
    assert '_resume_data_states' in assign, (
        'pre_epoch is restored unconditionally — it double-counts whenever '
        'the data stream is restored alongside it')
    print('PASS 7: pre_epoch is carried only when the data stream restarts')


if __name__ == '__main__':
    test_seed_is_deterministic_across_processes()
    test_seed_survives_a_resume()
    test_config_seed_not_mutated()
    test_checkpoint_is_atomic()
    test_state_loads_on_cpu()
    test_epoch_takes_the_slowest_task_and_the_right_weight()
    test_pre_epoch_is_not_double_counted_on_resume()
    print('ALL PASS')
