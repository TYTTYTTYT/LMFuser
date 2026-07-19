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


if __name__ == '__main__':
    test_mixed_kinds_across_ranks()
    print('ALL PASS')
