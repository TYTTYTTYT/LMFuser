"""Worker for test_gather.py — 4 ranks, mismatched key kinds."""
import sys, os, torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
dist.init_process_group('gloo')
r, W = dist.get_rank(), dist.get_world_size()
from lmfuser.utils import batch_all_gather
ok = []

# ① 只有一个 rank 有该 key
b = {'only0': torch.tensor([[float(r)]])} if r == 0 else {}
o = batch_all_gather(b)
ok.append(('单 rank 持有', o['only0'].flatten().tolist() == [0.0]))

# ② 除一个 rank 外都有
b = {} if r == 1 else {'most': torch.tensor([[float(r)]])}
o = batch_all_gather(b)
ok.append(('除一个外都有', sorted(o['most'].flatten().tolist()) == sorted([float(x) for x in range(W) if x != 1])))

# ③ 同一 key,一个 rank 是空 list,另一个非空
b = {'mix': [f'r{r}'] if r % 2 == 0 else []}
o = batch_all_gather(b)
ok.append(('空/非空 list 混合', sorted(o['mix']) == sorted([f'r{x}' for x in range(W) if x % 2 == 0])))

# ④ 非 float dtype
b = {'ints': torch.tensor([[r]], dtype=torch.int64)} if r != 2 else {}
o = batch_all_gather(b)
ok.append(('int64 + 缺席 rank', o['ints'].dtype == torch.int64))

# ⑤ 非字符串 key
b = {7: torch.tensor([[float(r)]]), ('t', 1): [r]}
o = batch_all_gather(b)
ok.append(('非字符串 key', 7 in o and ('t',1) in o))

if r == 0:
    for name, good in ok: print(f'  {"OK " if good else "!! "}{name}')
    print('GATHER_MULTI_OK' if all(g for _, g in ok) else 'SOME_FAILED')
dist.barrier(); dist.destroy_process_group()
