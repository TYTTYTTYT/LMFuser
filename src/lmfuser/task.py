from typing import Any, Callable
from collections.abc import Iterable, Iterator

import torch
from torch import nn
from lmfuser_data.interfaces import Batch, Row
from lmfuser_data.scanners import Scanner
from lmfuser_data import DataLoader, PyTorchDataLoader
from lmfuser_data.interfaces import SubclassTracer
from hyperargs import Conf, StrArg, FloatArg, IntArg, OptionArg, add_dependency, monitor_on

def scanner_type_list() -> list[str]:
    return list(Scanner.all_subclass_names())


class EmptyDataLoader:
    '''
        EmptyDataLoader is a class that provides empty data for running with no data requierment.
    '''
    def __init__(self, init_step: int = 0) -> None:
        self.init_step = init_step

    @property
    def epoch(self) -> int:
        return 0

    def __iter__(self) -> Iterator[Batch]:
        def it_wrap() -> Iterator[Batch]:
            while True:
                yield {'step': torch.tensor(self.init_step)}
                self.init_step += 1
        return it_wrap()


@add_dependency('num_train_data_path', 'train_data_path_list')
@add_dependency('num_train_data_path', 'train_data_weights')
@add_dependency('num_eval_data_path', 'eval_data_path_list')
@add_dependency('num_eval_data_path', 'eval_data_weights')
@add_dependency('num_test_data_path', 'test_data_path_list')
@add_dependency('num_test_data_path', 'test_data_weights')
class TaskBase(Conf, SubclassTracer):
    num_train_data_path = IntArg(1, min_value=0)
    train_data_path_list = [StrArg('Enther the path to the data file.')]
    train_data_weights = [FloatArg(1.0, min_value=0.0, max_value=1.0)]

    num_eval_data_path = IntArg(1, min_value=0)
    eval_data_path_list = [StrArg('Enther the path to the data file.')]
    eval_data_weights = [FloatArg(1.0, min_value=0.0, max_value=1.0)]

    num_test_data_path = IntArg(0, min_value=0)
    test_data_path_list: list[StrArg] = []
    test_data_weights: list[FloatArg] = []

    scanner_type = OptionArg(default='C4Scanner', option_fn=scanner_type_list)

    train_dataloader_type = OptionArg(default='single file', options=['single file', 'sharded', 'empty'])
    eval_dataloader_type = OptionArg(default='single file', options=['single file', 'sharded', 'empty'])
    test_dataloader_type = OptionArg(default='single file', options=['single file', 'sharded', 'empty'])

    _train_dataloader: DataLoader | None | PyTorchDataLoader | EmptyDataLoader = None
    _eval_dataloader: DataLoader | None | PyTorchDataLoader | EmptyDataLoader = None
    _test_dataloader: DataLoader | None | PyTorchDataLoader | EmptyDataLoader = None

    @monitor_on('num_train_data_path')
    def set_train_path_list(self) -> None:
        num = self.num_train_data_path.value()
        assert isinstance(num, int)
        if len(self.train_data_path_list) > num:
            self.train_data_path_list = self.train_data_path_list[:num]
            self.train_data_weights = self.train_data_weights[:num]
        elif len(self.train_data_path_list) < num:
            self.train_data_path_list += [StrArg('Enther the path to the data file.')] * (num - len(self.train_data_path_list))
            self.train_data_weights += [FloatArg(1.0, min_value=0.0, max_value=1.0)] * (num - len(self.train_data_weights))

    @monitor_on('num_eval_data_path')
    def set_eval_path_list(self) -> None:
        num = self.num_eval_data_path.value()
        assert isinstance(num, int)
        if len(self.eval_data_path_list) > num:
            self.eval_data_path_list = self.eval_data_path_list[:num]
            self.eval_data_weights = self.eval_data_weights[:num]
        elif len(self.eval_data_path_list) < num:
            self.eval_data_path_list += [StrArg('Enther the path to the data file.')] * (num - len(self.eval_data_path_list))
            self.eval_data_weights += [FloatArg(1.0, min_value=0.0, max_value=1.0)] * (num - len(self.eval_data_weights))

    @monitor_on('num_test_data_path')
    def set_test_path_list(self) -> None:
        num = self.num_test_data_path.value()
        assert isinstance(num, int)
        if len(self.test_data_path_list) > num:
            self.test_data_path_list = self.test_data_path_list[:num]
            self.test_data_path_list = self.test_data_path_list[:num]
        elif len(self.test_data_path_list) < num:
            self.test_data_path_list += [StrArg('Enther the path to the data file.')] * (num - len(self.test_data_path_list))
            self.test_data_weights += [FloatArg(1.0, min_value=0.0, max_value=1.0)] * (num - len(self.test_data_weights))

    def _get_train_dataloader(
        self,
        batch_size: int,
        seed: int,
        shuffle: bool,
        prefetch_factor: int,
        ignore_error: bool,
        qps: float | None,
        instruct_timeout: float,
        worker_timeout: float,
        num_workers: int,
        rank: int,
        world_size: int
    ) -> None | DataLoader | PyTorchDataLoader | EmptyDataLoader:
        if self.num_train_data_path.value() == 0:
            return None
        if self._train_dataloader is not None:
            return self._train_dataloader
        path_list = [p.value() for p in self.train_data_path_list]
        weight_list = [w.value() for w in self.train_data_weights]
        scanner_type = self.scanner_type.value()
        assert scanner_type is not None, 'scanner_type is None'

        dataloader_type = self.train_dataloader_type.value()
        assert dataloader_type in ('sharded', 'single file'), f'Unknown dataloader type: {dataloader_type}'

        if dataloader_type == 'sharded':
            self._train_dataloader = DataLoader(
                batch_size=batch_size,
                path_list=path_list, # type: ignore
                distributor_weights=weight_list, # type: ignore
                scanner_type=Scanner.get_subclass(scanner_type),
                seed=seed,
                shuffle=shuffle,
                pre_fetch_factor=prefetch_factor,
                ignore_error=ignore_error,
                qps=qps,
                instruct_timeout=instruct_timeout,
                worker_timeout=worker_timeout,
                num_workers=num_workers,
                map_fn=self.get_row_processor(),
                flow_fn=self.get_flow_processor(),
                batch_map_fn=self.get_batch_processor(),
                rank_idx=rank,
                num_ranks=world_size,
            )
        elif dataloader_type == 'single file':
            self._train_dataloader = PyTorchDataLoader(
                batch_size=batch_size,
                path_list=path_list, # type: ignore
                scanner_type=Scanner.get_subclass(scanner_type),
                seed=seed,
                shuffle=shuffle,
                pre_fetch_factor=prefetch_factor,
                num_workers=num_workers,
                num_ranks=world_size,
                rank_idx=rank,
                collate_fn=self.get_collate_fn(),
                drop_last=False
            )
        elif dataloader_type == 'empty':
            self._train_dataloader = EmptyDataLoader(init_step=0)
        else:
            raise ValueError(f'Unknown dataloader type: {dataloader_type}')

        return self._train_dataloader

    def _get_eval_dataloader(
        self,
        batch_size: int,
        seed: int,
        shuffle: bool,
        prefetch_factor: int,
        ignore_error: bool,
        qps: float | None,
        instruct_timeout: float,
        worker_timeout: float,
        num_workers: int,
        rank: int,
        world_size: int
    ) -> None | DataLoader | PyTorchDataLoader | EmptyDataLoader:
        if self.num_eval_data_path.value() == 0:
            return None
        if self._eval_dataloader is not None:
            return self._eval_dataloader
        path_list = [p.value() for p in self.eval_data_path_list]
        weight_list = [w.value() for w in self.eval_data_weights]
        scanner_type = self.scanner_type.value()
        assert scanner_type is not None, 'scanner_type is None'

        dataloader_type = self.eval_dataloader_type.value()
        assert dataloader_type is not None, 'dataloader_type is None'

        dataloader_type = self.eval_dataloader_type.value()
        assert dataloader_type in ('sharded', 'single file'), f'Unknown dataloader type: {dataloader_type}'

        if dataloader_type == 'sharded':
            self._eval_dataloader = DataLoader(
                batch_size=batch_size,
                path_list=path_list, # type: ignore
                distributor_weights=weight_list, # type: ignore
                scanner_type=Scanner.get_subclass(scanner_type),
                seed=seed,
                shuffle=shuffle,
                pre_fetch_factor=prefetch_factor,
                ignore_error=ignore_error,
                qps=qps,
                instruct_timeout=instruct_timeout,
                worker_timeout=worker_timeout,
                num_workers=num_workers,
                map_fn=self.get_row_processor(),
                flow_fn=self.get_flow_processor(),
                batch_map_fn=self.get_batch_processor(),
                rank_idx=rank,
                num_ranks=world_size,
            )
        elif dataloader_type == 'single file':
            self._eval_dataloader = PyTorchDataLoader(
                batch_size=batch_size,
                path_list=path_list, # type: ignore
                scanner_type=Scanner.get_subclass(scanner_type),
                seed=seed,
                shuffle=shuffle,
                pre_fetch_factor=prefetch_factor,
                num_workers=num_workers,
                num_ranks=world_size,
                rank_idx=rank,
                collate_fn=self.get_collate_fn(),
                drop_last=False
            )
        elif dataloader_type == 'empty':
            self._eval_dataloader = EmptyDataLoader(init_step=0)
        else:
            raise ValueError(f'Unknown dataloader type: {dataloader_type}')

        return self._eval_dataloader

    def _get_test_dataloader(
        self,
        batch_size: int,
        seed: int,
        shuffle: bool,
        prefetch_factor: int,
        ignore_error: bool,
        qps: float | None,
        instruct_timeout: float,
        worker_timeout: float,
        num_workers: int,
        rank: int,
        world_size: int
    ) -> None | DataLoader | PyTorchDataLoader | EmptyDataLoader:
        if self.num_test_data_path.value() == 0:
            return None
        if self._test_dataloader is not None:
            return self._test_dataloader
        path_list = [p.value() for p in self.test_data_path_list]
        weight_list = [w.value() for w in self.test_data_weights]
        scanner_type = self.scanner_type.value()
        assert scanner_type is not None, 'scanner_type is None'

        dataloader_type = self.test_dataloader_type.value()
        assert dataloader_type is not None, 'dataloader_type is None'

        dataloader_type = self.test_dataloader_type.value()
        assert dataloader_type in ('sharded', 'single file'), f'Unknown dataloader type: {dataloader_type}'

        if dataloader_type == 'sharded':
            self._test_dataloader = DataLoader(
                batch_size=batch_size,
                path_list=path_list, # type: ignore
                distributor_weights=weight_list, # type: ignore
                scanner_type=Scanner.get_subclass(scanner_type),
                seed=seed,
                shuffle=shuffle,
                pre_fetch_factor=prefetch_factor,
                ignore_error=ignore_error,
                qps=qps,
                instruct_timeout=instruct_timeout,
                worker_timeout=worker_timeout,
                num_workers=num_workers,
                map_fn=self.get_row_processor(),
                flow_fn=self.get_flow_processor(),
                batch_map_fn=self.get_batch_processor(),
                rank_idx=rank,
                num_ranks=world_size,
            )
        elif dataloader_type == 'single file':
            self._test_dataloader = PyTorchDataLoader(
                batch_size=batch_size,
                path_list=path_list, # type: ignore
                scanner_type=Scanner.get_subclass(scanner_type),
                seed=seed,
                shuffle=shuffle,
                pre_fetch_factor=prefetch_factor,
                num_workers=num_workers,
                num_ranks=world_size,
                rank_idx=rank,
                collate_fn=self.get_collate_fn(),
                drop_last=False
            )
        elif dataloader_type == 'empty':
            self._test_dataloader = EmptyDataLoader(init_step=0)
        else:
            raise ValueError(f'Unknown dataloader type: {dataloader_type}')

        return self._test_dataloader

    def train_step(
        self, model: nn.Module,
        batch: Batch,
        step: int,
        device: Any,
        acc_step: int,
        **kwargs: Any
    ) -> Batch | torch.Tensor:
        raise NotImplementedError('Please implement this method in child class')

    def eval_step(
        self,
        model: nn.Module,
        batch: Batch,
        step: int,
        device: Any,
        **kwargs: Any
    ) -> dict[str, list[Any]]:
        raise NotImplementedError('Please implement this method in child class')

    def cal_dev_metric(self, eval_outputs: dict[str, list[Any]]) -> dict[str, Any]:
        raise NotImplementedError('Please implement this method in child class')

    def set_test_model_path(self, checkpoints_path: str, dev_metrics: dict[str, list[dict[str, Any]]]) -> str | int | None:
        '''Set the path of the test model based on the dev metrics.
        Args:
            checkpoints_path (str): The path to the checkpoints directory.
            dev_metrics (dict[str, list[dict[str, Any]]]): The development metrics for each task.

        Returns:
            str | int | None: The path or the checkpoint step to the test model, or None to use the last step model.
        '''
        return None

    def get_row_processor(self) -> Callable[[Row], Row] | None:
        return None

    def get_flow_processor(self) -> Callable[[Iterable[Row]], Iterable[Row]] | None:
        return None

    def get_batch_processor(self) -> Callable[[Batch], Batch] | None:
        return None

    def get_collate_fn(self) -> Callable[[list[Row]], Batch] | None:
        return None


class Task(TaskBase):
    pass

def task_list() -> list[str]:
    return list(TaskBase.all_subclass_names())


@add_dependency('conf', 'task_name')
class TaskSelector(Conf):
    task_name = OptionArg(default='Task', option_fn=task_list)
    conf: TaskBase = Task()

    @monitor_on('task_name')
    def change_conf(self) -> None:
        name = self.task_name.value()
        if name is None:
            self.conf = Task()

        elif name != self.conf.__class__.__name__:
            self.conf = TaskBase.all_subclass_map()[name]()


@add_dependency('num_tasks', 'tasks')
class Tasks(Conf):
    num_tasks = IntArg(1, min_value=1)
    tasks = [TaskSelector()]
    task_weights = [FloatArg(1.0, min_value=0.0, max_value=1.0)]

    @monitor_on('num_tasks')
    def change_task_list(self) -> None:
        num = self.num_tasks.value()
        assert num is not None, 'num_tasks is None'

        if len(self.tasks) > num:
            self.tasks = self.tasks[:num]
            self.task_weights = self.task_weights[:num]
        elif len(self.tasks) < num:
            self.tasks += [TaskSelector() for _ in range(num - len(self.tasks))]
            self.task_weights += [
                FloatArg(1.0, min_value=0.0, max_value=1.0) for _ in range(num - len(self.task_weights))
            ]

    def get_train_dataloaders(
        self,
        batch_size: int,
        seed: int,
        shuffle: bool,
        prefetch_factor: int,
        ignore_error: bool,
        qps: float | None,
        instruct_timeout: float,
        worker_timeout: float,
        num_workers: int,
        rank: int,
        world_size: int
    ) -> list[DataLoader | None | PyTorchDataLoader | EmptyDataLoader]:
        return [
            task.conf._get_train_dataloader(
                batch_size=batch_size,
                seed=seed,
                shuffle=shuffle,
                prefetch_factor=prefetch_factor,
                ignore_error=ignore_error,
                qps=qps,
                instruct_timeout=instruct_timeout,
                worker_timeout=worker_timeout,
                num_workers=num_workers,
                rank=rank,
                world_size=world_size,
            )
            for task in self.tasks
        ]

    def get_eval_dataloaders(
        self,
        batch_size: int,
        seed: int,
        shuffle: bool,
        prefetch_factor: int,
        ignore_error: bool,
        qps: float | None,
        instruct_timeout: float,
        worker_timeout: float,
        num_workers: int,
        rank: int,
        world_size: int
    ) -> list[DataLoader | None | PyTorchDataLoader | EmptyDataLoader]:
        return [
            task.conf._get_eval_dataloader(
                batch_size=batch_size,
                seed=seed,
                shuffle=shuffle,
                prefetch_factor=prefetch_factor,
                ignore_error=ignore_error,
                qps=qps,
                instruct_timeout=instruct_timeout,
                worker_timeout=worker_timeout,
                num_workers=num_workers,
                rank=rank,
                world_size=world_size,
            )
            for task in self.tasks
        ]

    def get_test_dataloaders(
        self,
        batch_size: int,
        seed: int,
        shuffle: bool,
        prefetch_factor: int,
        ignore_error: bool,
        qps: float | None,
        instruct_timeout: float,
        worker_timeout: float,
        num_workers: int,
        rank: int,
        world_size: int
    ) -> list[DataLoader | None | PyTorchDataLoader | EmptyDataLoader]:
        return [
            task.conf._get_test_dataloader(
                batch_size=batch_size,
                seed=seed,
                shuffle=shuffle,
                prefetch_factor=prefetch_factor,
                ignore_error=ignore_error,
                qps=qps,
                instruct_timeout=instruct_timeout,
                worker_timeout=worker_timeout,
                num_workers=num_workers,
                rank=rank,
                world_size=world_size,
            )
            for task in self.tasks
        ]
