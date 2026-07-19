import sys, torch, torch.distributed as dist
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
dist.init_process_group('gloo')
from lmfuser.utils import dist_avg, batch_all_gather
r = dist.get_rank()
print(f'rank{r}: dist_avg(float) =', dist_avg(float(r)))
print(f'rank{r}: dist_avg(tensor) =', dist_avg(torch.tensor([float(r)])).tolist())
o = batch_all_gather({'loss': torch.tensor(float(r))})   # 0 维张量
if r == 0: print('0 维张量 gather ->', o['loss'].tolist())
dist.barrier(); dist.destroy_process_group()
if r == 0: print('CPU_MULTIRANK_OK')
