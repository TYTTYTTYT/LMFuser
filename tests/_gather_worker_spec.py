import sys, torch, torch.distributed as dist
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
dist.init_process_group('gloo')
r = dist.get_rank()
from lmfuser.utils import batch_all_gather
# ① dtype 分歧:必须两个 rank 一起报错,而不是位重解释
b = {'x': torch.tensor([1,2], dtype=torch.int64)} if r==0 else {'x': torch.tensor([3.,4.])}
try:
    batch_all_gather(b); print(f'rank{r}: !! 未报错')
except TypeError as e:
    print(f'rank{r}: OK 一起报错 -> {str(e)[:60]}')
dist.barrier()
# ② identity-hash key:必须报错而不是静默丢数据
class K:
    def __init__(s, n): s.n = n
    def __hash__(s): return id(s)
    def __eq__(s, o): return s is o
try:
    batch_all_gather({K('m'): torch.tensor([[float(r)]])}); print(f'rank{r}: !! 静默通过')
except TypeError as e:
    print(f'rank{r}: OK key 往返检查 -> {str(e)[:55]}')
dist.destroy_process_group()

if r == 0: print('GATHER_SPEC_OK')
