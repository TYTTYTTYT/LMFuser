"""Worker for test_gather.py — run under torchrun with 2 ranks."""
import os, sys, torch, torch.distributed as dist
import os as _os
sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), '..', 'src'))
dist.init_process_group('gloo')
rank = dist.get_rank()
from lmfuser.utils import batch_all_gather

# rank0 有张量 + 字符串列表;rank1 是"空 eval 分片",什么都没有
if rank == 0:
    batch = {'loss': torch.tensor([[1.0],[2.0]]), 'name': ['a', 'b'], 'plain': 'abc'}
else:
    batch = {}
out = batch_all_gather(batch)
if rank == 0:
    print('keys:', sorted(out))
    print('loss :', out['loss'].flatten().tolist())
    print('name :', out['name'])
    print('plain:', out['plain'], ' <- 未被拆成字符')
dist.barrier(); dist.destroy_process_group()
assert out['plain'] == ['abc'], out['plain']
assert out['loss'].flatten().tolist() == [1.0, 2.0], out['loss']
assert out['name'] == ['a', 'b'], out['name']
if rank == 0:
    print('GATHER_MIXED_KINDS_OK')
