"""Collective safety for batch_all_gather (0.4.5).

Ranks can legitimately hold different keys — a task that emits a metric only
on some steps, or a rank whose eval slice is empty. Agreeing on the key set is
not enough: the collective was also chosen from the LOCAL value, so a rank
holding tensors called all_gather while a rank missing that key called
all_gather_object. Same call count, different collective, silent hang until
the NCCL watchdog fires.

Run:  python tests/test_gather.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def test_mixed_kinds_across_ranks() -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    r = subprocess.run(
        [sys.executable, '-m', 'torch.distributed.run', '--standalone',
         '--nnodes=1', '--nproc_per_node=2', os.path.join(HERE, '_gather_worker.py')],
        capture_output=True, text=True, timeout=180, env=env,
    )
    assert 'GATHER_MIXED_KINDS_OK' in r.stdout, (
        'ranks holding different key kinds did not complete the gather:\n'
        + r.stdout[-2000:] + r.stderr[-2000:])
    print('PASS: mixed key kinds across ranks complete without hanging')


def _run(worker: str, nproc: int, marker: str) -> None:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    r = subprocess.run(
        [sys.executable, '-m', 'torch.distributed.run', '--standalone',
         '--nnodes=1', f'--nproc_per_node={nproc}', os.path.join(HERE, worker)],
        capture_output=True, text=True, timeout=240, env=env,
    )
    assert marker in r.stdout, (
        f'{worker} did not reach {marker}:\n' + r.stdout[-2000:] + r.stderr[-2000:])


def test_missing_and_empty_keys_across_four_ranks() -> None:
    """A key on one rank only, on all but one, an empty list beside a full one,
    a non-float dtype with an absent rank, and non-str keys."""
    _run('_gather_worker_multi.py', 4, 'GATHER_MULTI_OK')
    print('PASS: four ranks with missing / empty / non-str keys all complete')


def test_ranks_disagreeing_on_a_key_kind() -> None:
    """Ranks reporting different KINDS for one key must still finish.

    The kind negotiation picks one, and the rank whose value does not match it
    used to fall into torch.cat on a list of strings — it raised and left,
    while every other rank waited in the collective it had already entered.
    """
    _run('_gather_worker_conflict.py', 2, 'GATHER_CONFLICT_OK')
    print('PASS: a key-kind disagreement completes instead of hanging')


def test_ranks_disagreeing_on_dtype_or_key_identity() -> None:
    """Disagreements must surface on EVERY rank at once.

    Taking one rank's spec silently reinterprets another's bytes — an int64
    rank gathering a float32 rank's rows produced 1077936128 for 3.0, which
    reads like a plausible metric. And a key whose hash is its identity comes
    back from pickle as a different object, so `batch.get(k)` missed on the
    rank that held the data and every rank contributed nothing: the data
    vanished with no error at all.
    """
    _run('_gather_worker_spec.py', 2, 'GATHER_SPEC_OK')
    print('PASS: dtype disagreement and identity-hash keys raise on all ranks')


def test_cpu_multi_rank() -> None:
    """dist_avg and the gather must work without CUDA.

    get_default_device() reports -1 for CPU, which torch rejects as a device
    index, so multi-rank CPU training died at its first logging step.
    """
    _run('_gather_worker_cpu.py', 2, 'CPU_MULTIRANK_OK')
    print('PASS: dist_avg and batch_all_gather work on CPU across ranks')


if __name__ == '__main__':
    test_mixed_kinds_across_ranks()
    test_missing_and_empty_keys_across_four_ranks()
    test_ranks_disagreeing_on_a_key_kind()
    test_ranks_disagreeing_on_dtype_or_key_identity()
    test_cpu_multi_rank()
    print('ALL PASS')
