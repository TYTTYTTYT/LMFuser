"""Worker for test_gather.py — ranks disagreeing on a key's kind."""
import sys, torch, torch.distributed as dist
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
dist.init_process_group('gloo')
r = dist.get_rank()
from lmfuser.utils import batch_all_gather
# rank0 给张量,rank1 给对象列表 —— 同一个 key,种类真正冲突
batch = {'conflict': torch.tensor([[1.0]])} if r == 0 else {'conflict': ['x']}
try:
    out = batch_all_gather(batch)
    if r == 0:
        print('完成,未死锁; 结果类型:', type(out['conflict']).__name__)
except Exception as e:
    if r == 0: print(f'以异常结束(可接受,只要不挂): {type(e).__name__}: {str(e)[:70]}')
dist.barrier(); dist.destroy_process_group()
if r == 0: print('GATHER_CONFLICT_OK')
