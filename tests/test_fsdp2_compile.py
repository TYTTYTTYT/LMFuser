"""FSDP2Wrapper + compile_mode coverage (synthetic, single GPU).

Run:  torchrun --nproc_per_node=1 tests/test_fsdp2_compile.py

Covers, per compile_mode in {disable, default, reduce-overhead}:
  * forward/backward/optimizer step run under fully_shard;
  * parameters are NOT duplicated by compilation (the OptimizedModule must not
    be registered as a submodule);
  * 'reduce-overhead' downgrades to 'default' with a critical log.
"""
import logging
import os
import sys

import torch
from torch import nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lmfuser.runners.ddp_runner import FSDP2Wrapper   # noqa: E402


def build_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64),
    )


def run_mode(mode: str) -> None:
    model = build_model().cuda()
    n_params_before = len(list(model.parameters()))
    fully_shard(model)

    records: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda r: records.append(r.getMessage())   # type: ignore
    logging.getLogger('lmfuser.runners.ddp_runner').addHandler(handler)

    wrapper = FSDP2Wrapper(model, compile_mode=mode)

    # ---- parameter registration must not be duplicated by compile ----
    n_params_after = len(list(wrapper.parameters()))
    assert n_params_after == n_params_before, \
        f'[{mode}] parameter duplication: {n_params_before} -> {n_params_after}'
    opt = torch.optim.AdamW(wrapper.parameters(), lr=1e-3)   # raises on duplicates

    # ---- train a few steps ----
    for _ in range(3):
        x = torch.randn(8, 64, device='cuda')
        loss = wrapper(x).square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert torch.isfinite(loss).item(), f'[{mode}] non-finite loss'

    # ---- downgrade warning ----
    if mode == 'reduce-overhead':
        assert any('downgrading' in m for m in records), \
            '[reduce-overhead] downgrade warning not logged'

    print(f'PASS fsdp2 compile_mode={mode} (loss={loss.item():.4f})')


if __name__ == '__main__':
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(0)
    for mode in ('disable', 'default', 'reduce-overhead'):
        run_mode(mode)
    dist.destroy_process_group()
    print('FSDP2_COMPILE_TESTS_PASS')
