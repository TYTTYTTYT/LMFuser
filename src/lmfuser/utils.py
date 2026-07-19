import os
from typing import Any, overload, Union, Optional, Literal
from random import Random

import torch
from torch import distributed as dist
from torch.utils.data import get_worker_info
from torch import Tensor
import atexit

from typing import TypeVar
import random
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')

def get_world_size() -> int:
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE'])
    else:
        return 1

def get_global_rank() -> int:
    if 'RANK' in os.environ:
        return int(os.environ['RANK'])
    else:
        return 0

def get_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    else:
        # Provide a default or handle the case of non-distributed training
        return 0

def dist_init() -> None:
    if dist.is_initialized():
        return

    device_type = get_default_device_type()

    if device_type == 'cuda':
        import datetime as _dt
        # 1h timeout: a transient data stall on one rank (e.g. a shard
        # download hang) must not nuke the whole DDP group (default 10min
        # killed a pretrain run mid-flight)
        dist.init_process_group(backend='nccl', timeout=_dt.timedelta(seconds=3600))
        torch.cuda.set_device(get_local_rank())
    elif device_type == 'npu':
        dist.init_process_group(backend='hccl')
    else:
        dist.init_process_group(backend='mpi')

    atexit.register(dist.destroy_process_group)

def weighted_random_choice(
    elements: list[T], 
    probabilities: list[float],
    rand: Optional[Random] = None
) -> T:
    """
    Choose a random element from a list based on specified probabilities.

    Args:
        elements (List[T]): A list of elements to choose from.
        probabilities (List[float]): A list of probabilities associated with each element. Must sum to 1.
        rand (Optional[Random]): The random number generator.

    Returns:
        T: A randomly chosen element from the list.
    """
    if len(elements) != len(probabilities):
        raise ValueError("Elements and probabilities must have the same length.")
    if not all(0 <= p <= 1 for p in probabilities):
        raise ValueError("Probabilities must be non-negative and non-greater than 1.")
    if not abs(sum(probabilities) - 1.0) < 1e-6:
        raise ValueError("Probabilities must sum to 1.")

    if rand is None:
        index = random.choices(list(range(len(elements))), probabilities)[0]
    else:
        index = rand.choices(list(range(len(elements))), probabilities)[0]

    return elements[index]

def partition_list(lst: list[T], num_shards: int, index: int) -> list[T]:
    # Ensure the number of shards is positive and index is valid
    if num_shards <= 0 or index >= num_shards or index < 0:
        raise ValueError("Invalid number of shards or index, "
                         f"number of shards: {num_shards};"
                         f"index: {index}.")

    # Calculate the size of each shard
    shard_size = len(lst) // num_shards
    remainder = len(lst) % num_shards
    
    # Calculate the start and end indices for the partition
    start = index * shard_size + min(index, remainder)
    end = (index + 1) * shard_size + min(index + 1, remainder)
    
    return lst[start:end]    

DEVICE_TYPE: Optional[Literal['cuda', 'npu', 'cpu']] = None
def get_default_device_type() -> Literal['cuda', 'npu', 'cpu']:
    global DEVICE_TYPE
    if DEVICE_TYPE is not None:
        return DEVICE_TYPE

    device = os.environ.get('HURRICANE_DEVICE')
    if device is not None:
        DEVICE_TYPE = device # type: ignore
        return DEVICE_TYPE # type: ignore

    if torch.cuda.is_available():
        DEVICE_TYPE = 'cuda'
        return DEVICE_TYPE

    try:
        __import__('torch_npu')
        if torch_npu.npu.is_available(): # type: ignore
            DEVICE_TYPE = 'npu'
            return DEVICE_TYPE
    except ImportError:
        ...

    DEVICE_TYPE = 'cpu'

    return DEVICE_TYPE

DEVICE: Optional[str] = None
def get_default_device() -> str | int:
    """
    Get the default device for the current process.
    """
    global DEVICE
    if DEVICE is not None:
        return DEVICE

    device_type = get_default_device_type()
    if device_type == 'cpu':
        return -1
    if device_type == 'cuda':
        return get_local_rank()

    return f'{device_type}:{get_local_rank()}'

def torch_device() -> Union[str, int]:
    """`get_default_device()` reports -1 for CPU, which torch rejects as a
    device index. Everything that actually constructs a tensor needs this."""
    dev = get_default_device()
    return 'cpu' if dev == -1 else dev


@overload
def dist_avg(value: Tensor) -> Tensor: ...
@overload
def dist_avg(value: int) -> float: ...
@overload
def dist_avg(value: float) -> float: ...
def dist_avg(value: Union[torch.Tensor, int, float]) -> Union[torch.Tensor, float]:
    if not dist.is_initialized():
        return value
    
    if isinstance(value, torch.Tensor):
        return_tensor = True
        value = value.to(torch_device())
    else:
        return_tensor = False
        value = torch.tensor(value, device=torch_device(), dtype=torch.float32)
        
    dist.all_reduce(value, dist.ReduceOp.SUM)
    dist.barrier()

    if return_tensor:
        return value / dist.get_world_size()

    return value.item() / dist.get_world_size()


def gather_object(local_object: T) -> list[T]:
    if not dist.is_initialized():
        return [local_object]
    world_size = dist.get_world_size()

    gathered = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_object)

    if isinstance(gathered[0], list):
        results = []
        for l in gathered:
            results += l # type: ignore
        gathered = results
    
    return gathered # type: ignore

def tensor_all_gather(tensor: Tensor) -> Tensor:
    """Gather along dim 0 across ranks, allowing different lengths.

    `empty_like` per rank assumed every rank held the same number of rows.
    That holds for a padded DistributedSampler and nowhere else: with uneven
    eval counts gloo aborts the process and NCCL does not validate sizes at
    all, so it silently truncates or corrupts. Sizes are exchanged first, then
    each rank's slice is padded to the largest and trimmed back after.
    """
    if get_world_size() <= 1:
        return tensor
    if tensor.dim() == 0:
        # concatenating along dim 0 needs a dim 0 to exist
        tensor = tensor.reshape(1)
    device = torch_device()
    src = tensor.contiguous().to(device)

    counts = torch.zeros(dist.get_world_size(), dtype=torch.long, device=device)
    counts[dist.get_rank()] = src.shape[0]
    dist.all_reduce(counts)
    longest = int(counts.max().item())

    if src.shape[0] < longest:
        pad = torch.zeros((longest - src.shape[0], *src.shape[1:]),
                          dtype=src.dtype, device=device)
        padded = torch.cat([src, pad], dim=0)
    else:
        padded = src

    results = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(results, padded)

    trimmed = [r[:int(counts[i].item())] for i, r in enumerate(results)]
    return torch.cat(trimmed, dim=0).to(tensor.device)

def batch_all_gather(batch: dict[str, Any]) -> dict[str, Any]:
    """把各个rank的同一批batch数据汇总到rank0，方便计算metric或者刷库等等。
    不是分布式的话原样返回

    Args:
        batch (Dict[str, Union[Tensor, List[Any]]]): 一个batch的数据，注意一个key对应一个list

    Returns:
        Dict[Dict[str, Union[Tensor, List[Any]]]]: 汇总后的batch
    """
    if not dist.is_initialized():
        return batch

    # Agree on the key set AND on what kind of value each key holds, before
    # issuing anything.
    #
    # This used to loop over the local dict, so a rank with a different key set
    # — a task emitting a metric only on some steps, or a rank whose eval slice
    # is empty — made a different NUMBER of collective calls and the job hung.
    # Agreeing only on the keys is not enough either: the branch below is
    # chosen from the local value, so a rank holding tensors would call
    # all_gather while a rank missing that key called all_gather_object. Same
    # count, different collective, same hang.
    local: dict[Any, tuple] = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            local[k] = ('tensor', tuple(v.shape[1:]), str(v.dtype))
        elif isinstance(v, (list, tuple)) and len(v) > 0 and isinstance(v[0], Tensor):
            local[k] = ('tensor_list', tuple(v[0].shape[1:]), str(v[0].dtype))
        else:
            local[k] = ('object', (), '')

    per_rank: list[Any] = [None] * dist.get_world_size()   # type: ignore[list-item]
    dist.all_gather_object(per_rank, local)

    kinds: dict[Any, tuple] = {}
    disagree: dict[Any, set] = {}
    for rank_map in per_rank:
        for k, spec in (rank_map or {}).items():
            # 'object' is also what a rank reports for an empty list, so a
            # concrete kind from any rank wins
            if k not in kinds or (kinds[k][0] == 'object' and spec[0] != 'object'):
                if k in kinds and kinds[k][0] != 'object' and kinds[k] != spec:
                    disagree.setdefault(k, {kinds[k]}).add(spec)
                kinds[k] = spec
            elif spec[0] != 'object' and spec != kinds[k]:
                disagree.setdefault(k, {kinds[k]}).add(spec)
    if disagree:
        # Every rank computes this from the same gathered data, so every rank
        # raises together — a one-sided failure here is what hangs the others.
        # Silently taking one rank's spec reinterprets another's bytes: an
        # int64 rank gathering a float32 rank's rows produced 1077936128 for
        # 3.0, which reads like a plausible metric.
        raise TypeError(
            'ranks disagree about what these batch keys hold: '
            + '; '.join(f'{k!r}: {sorted(v)}' for k, v in disagree.items())
        )

    # Keys travel through pickle, so a key whose hash is its identity comes
    # back as a DIFFERENT object and `batch.get(k)` misses on the very rank
    # that holds the data — every rank then contributes nothing and the data
    # vanishes with no error at all.
    missing = [k for k in batch if k not in kinds]
    if missing:
        raise TypeError(
            f'batch keys {missing!r} do not survive a pickle round-trip by '
            f'value (identity-based hash or eq?) — gathering them would '
            f'silently discard every rank\'s data'
        )

    gathered = {}
    for k in sorted(kinds, key=repr):
        kind, shape_tail, dtype_str = kinds[k]
        v = batch.get(k)
        if kind in ('tensor', 'tensor_list'):
            if isinstance(v, Tensor):
                local_t = v
            elif (isinstance(v, (list, tuple)) and len(v) > 0
                  and isinstance(v[0], Tensor)):
                local_t = torch.cat(list(v), dim=0)
            else:
                if v is not None and not (isinstance(v, (list, tuple)) and len(v) == 0):
                    # ranks disagree about what this key holds. Contributing
                    # nothing keeps this rank in the same collective as the
                    # others; raising here would leave them waiting on a peer
                    # that has already gone.
                    logger.warning(
                        f"rank {get_global_rank()} holds {type(v).__name__} for "
                        f"key {k!r} while the batch agreed on {kind}; its value "
                        f"is dropped from the gather"
                    )
                # this rank has nothing for this key: contribute zero rows so
                # it still takes part in the same collective
                dtype = getattr(torch, dtype_str.replace('torch.', ''), None)
                if dtype is None:
                    # never guess: a wrong dtype here is gathered into the
                    # metric as plausible-looking numbers
                    raise TypeError(
                        f'cannot reconstruct dtype {dtype_str!r} for key {k!r} '
                        f'while gathering from a rank that has no data for it'
                    )
                local_t = torch.empty((0, *shape_tail), dtype=dtype,
                                      device=torch_device())
            gathered[k] = tensor_all_gather(local_t)
        else:
            if v is None:
                payload: list = []
            elif isinstance(v, (list, tuple)):
                payload = list(v)
            else:
                payload = [v]        # never explode a str into characters
            gathered[k] = gather_object(payload)

    return gathered

def cal_acc_num(batch_size: int, sub_batch_size: int, world_size: int) -> int:
    acc_num = batch_size // sub_batch_size // world_size
    an = batch_size / sub_batch_size / world_size
    if abs(acc_num - an) > 1e-8:
        raise ValueError(
            f'Batchsize无法被平分！请根据显卡数合理设置“BatchSize”与“SubBatchSize”'
        )
    return acc_num


class MethodOverideChecker:
    def is_overridden(self, method_name: str) -> bool:
        """
        Check if the method `method_name` is overridden in this instance's class
        compared to the Parent class.
        """
        cls = self.__class__
        for parent in self.__class__.__bases__:
            if not hasattr(parent, method_name):
                continue
            return getattr(cls, method_name, None) is not getattr(parent, method_name, None)

        raise ValueError(f'No such method {method_name} in parent classes')
