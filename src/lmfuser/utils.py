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
        dist.init_process_group(backend='nccl')
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
        value = value.to(get_default_device())
    else:
        return_tensor = False
        value = torch.tensor(value, device=get_default_device(), dtype=torch.float32)
        
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
    if get_world_size() <= 1:
        return tensor
    results = [torch.empty_like(tensor, device=get_default_device()) for _ in range(dist.get_world_size())]

    dist.all_gather(results, tensor.contiguous().to(get_default_device()))

    return torch.cat(results, dim=0).to(tensor.device)

def batch_all_gather(batch: dict[str, Any]) -> dict[str, Any]:
    """把各个rank的同一批batch数据汇总到rank0，方便计算metric或者刷库等等。
    不是分布式的话原样返回

    Args:
        batch (Dict[str, Union[Tensor, List[Any]]]): 一个batch的数据，注意一个key对应一个list

    Returns:
        Dict[Dict[str, Union[Tensor, List[Any]]]]: 汇总后的batch
    """
    logger.info(f'Begin all gather on rank {get_global_rank()}')
    if not dist.is_initialized():
        return batch

    gathered = {}
    for k, v_list in batch.items():
        logger.info(f'Begin gather key {k} on rank {get_global_rank()}')
        if isinstance(v_list, list):
            logger.info(f'Key {k} with length {len(v_list)} on rank {get_global_rank()}')
        logger.info(f'after barrier on rank {get_global_rank()}')
        if isinstance(v_list, Tensor):
            gathered[k] = tensor_all_gather(v_list)
        else:
            if isinstance(v_list[0], Tensor):
                v_tensor = torch.cat(v_list, dim=0)
                gathered[k] = tensor_all_gather(v_tensor)
            else:
                gathered[k] = gather_object(v_list)

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
