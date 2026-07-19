"""Rejections must be symmetric: every rank raises, none is left in a collective.

Two ways a rank used to fail alone while its peers waited:

  ragged        one key holds a list of tensors that disagree past dim 0. The
                negotiated spec was sampled from v[0], so every rank AGREED on
                it and only the holder hit torch.cat's shape error.
  onesided_key  a key whose hash is its identity, held by ONE rank. The
                round-trip check read the local batch, so only that rank saw a
                problem to raise about.

Both are checked before the collective now, and the verdict travels inside it.
"""
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lmfuser.utils import batch_all_gather  # noqa: E402


class IdentityKey:
    """Hashes and compares by identity, so pickle returns a different object."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        return f'IdentityKey({self.name!r})'


def _expect_raises_everywhere(batch: dict, label: str) -> None:
    rank = dist.get_rank()
    try:
        batch_all_gather(batch)
        raised = ''
    except TypeError as e:
        raised = str(e)
    except Exception as e:  # pragma: no cover - a wrong exception is a failure
        raised = f'!!{type(e).__name__}: {e}'

    verdict = [None] * dist.get_world_size()
    dist.all_gather_object(verdict, raised)
    assert all(v for v in verdict), (
        f'{label}: not every rank raised — {[bool(v) for v in verdict]}. '
        f'A rank that returns normally here is a rank still inside a '
        f'collective its peers have already left.')
    assert all(not v.startswith('!!') for v in verdict), \
        f'{label}: wrong exception type somewhere: {verdict}'
    assert len(set(verdict)) == 1, \
        f'{label}: ranks raised DIFFERENT messages: {verdict}'
    if rank == 0:
        print(f'{label}: all {len(verdict)} ranks raised the same TypeError')


def main() -> None:
    dist.init_process_group('gloo')
    rank = dist.get_rank()

    # ragged: only rank 0 holds the unconcatenable list
    _expect_raises_everywhere(
        {'x': [torch.zeros(2, 4), torch.zeros(3, 9)]} if rank == 0
        else {'x': [torch.zeros(2, 4)]},
        'ragged tensor list on one rank',
    )

    # identity-hash key: only rank 0 holds it
    _expect_raises_everywhere(
        {IdentityKey('a'): torch.zeros(2, 3)} if rank == 0
        else {'fine': torch.zeros(2, 3)},
        'identity-hash key on one rank',
    )

    # torch.cat's preconditions are not two rules. Each of these passed a
    # trailing-dims-and-dtype check, was agreed on by every rank, and then
    # raised at the cat AFTER the collective.
    _expect_raises_everywhere(
        {'x': [torch.tensor(1.0), torch.tensor(2.0)]} if rank == 0
        else {'x': [torch.zeros(2)]},
        '0-dim tensors in a list',
    )
    _expect_raises_everywhere(
        {'x': [torch.zeros(2, 2).to_sparse(), torch.zeros(2, 2)]} if rank == 0
        else {'x': [torch.zeros(2, 2)]},
        'sparse beside dense',
    )

    # ... and it accepts more than those two rules allow. cat legacy-skips
    # shape-(0,) tensors, so rejecting this batch was a false positive.
    out = batch_all_gather({'x': [torch.zeros(0), torch.zeros(2, 3)]})
    assert tuple(out['x'].shape) == (2 * dist.get_world_size(), 3), out['x'].shape
    for label, value in (
        ('1-D of differing length', [torch.zeros(3), torch.zeros(5)]),
        ('0-row beside rows', [torch.zeros(0, 3), torch.zeros(2, 3)]),
        ('single element', [torch.zeros(2, 3)]),
    ):
        batch_all_gather({'x': value})   # must not raise
    if rank == 0:
        print('legitimate tensor lists still gather (no false positives)')

    # a key that cannot be pickled at all, and one whose __eq__ raises: both
    # used to escape from the pre-collective check itself, on one rank.
    class Unpicklable:
        def __hash__(self) -> int:
            return 7

        def __eq__(self, other: object) -> bool:
            return isinstance(other, Unpicklable)

        def __reduce__(self):
            raise TypeError('this key cannot be pickled')

    class BadEq:
        def __hash__(self) -> int:
            return 9

        def __eq__(self, other: object) -> bool:
            raise ValueError('__eq__ exploded')

    _expect_raises_everywhere(
        {Unpicklable(): torch.zeros(2, 3)} if rank == 0
        else {'fine': torch.zeros(2, 3)},
        'unpicklable key on one rank',
    )
    _expect_raises_everywhere(
        {BadEq(): torch.zeros(2, 3)} if rank == 0
        else {'fine': torch.zeros(2, 3)},
        'key whose __eq__ raises, on one rank',
    )

    # a healthy batch must still gather, and to the same result everywhere
    out = batch_all_gather({'x': torch.full((2, 3), float(rank)),
                            'y': [f'r{rank}']})
    assert tuple(out['x'].shape) == (2 * dist.get_world_size(), 3), out['x'].shape
    assert out['y'] == [f'r{r}' for r in range(dist.get_world_size())], out['y']

    if rank == 0:
        print('healthy batch still gathers correctly')
        print('GATHER_SYMMETRY_OK')

    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
