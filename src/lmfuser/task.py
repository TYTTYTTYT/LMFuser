from abc import ABC
from typing import TypeVar, Any, Callable
from collections.abc import Iterable

import torch
from torch import nn
from lmfuser_data.interfaces import Batch, Row
from lmfuser_data.scanners import Scanner
from lmfuser_data import DataLoader
from lmfuser_data.interfaces import SubclassTracer
from hyperargs import Conf, StrArg, FloatArg, IntArg, OptionArg, add_dependency, monitor_on


def scanner_type_list() -> list[str]:
    return list(Scanner.all_subclass_names())


@add_dependency('num_train_data_path', 'train_data_path_list')
@add_dependency('num_train_data_path', 'train_data_weights')
class TaskBase(Conf, SubclassTracer):
    num_train_data_path = IntArg(1, min_value=0)
    train_data_path_list = [StrArg('Enther the path to the data file.')]
    train_data_weights = [FloatArg(1.0, min_value=0.0, max_value=1.0)]

    num_eval_data_path = IntArg(1, min_value=0)
    eval_data_path_list = [StrArg('Enther the path to the data file.')]
    eval_data_weights = [FloatArg(1.0, min_value=0.0, max_value=1.0)]

    scanner_type = OptionArg(default='C4Scanner', option_fn=scanner_type_list)

    _train_dataloader: DataLoader | None = None
    _eval_dataloader: DataLoader | None = None

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
            self.eval_data_path_list += [StrArg('Enther the path to the data file.')] * (num - len(self.train_data_path_list))
            self.eval_data_weights += [FloatArg(1.0, min_value=0.0, max_value=1.0)] * (num - len(self.train_data_weights))

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
    ) -> DataLoader | None:
        if self.num_train_data_path.value() == 0:
            return None
        if self._train_dataloader is not None:
            return self._train_dataloader
        path_list = [p.value() for p in self.train_data_path_list]
        weight_list = [w.value() for w in self.train_data_weights]
        scanner_type = self.scanner_type.value()
        assert scanner_type is not None, 'scanner_type is None'

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
    ) -> DataLoader | None:
        if self.num_eval_data_path.value() == 0:
            return None
        if self._eval_dataloader is not None:
            return self._eval_dataloader
        path_list = [p.value() for p in self.eval_data_path_list]
        weight_list = [w.value() for w in self.eval_data_weights]
        scanner_type = self.scanner_type.value()
        assert scanner_type is not None, 'scanner_type is None'
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

        return self._eval_dataloader

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

    def get_row_processor(self) -> Callable[[Row], Row] | None:
        return None

    def get_flow_processor(self) -> Callable[[Iterable[Row]], Iterable[Row]] | None:
        return None

    def get_batch_processor(self) -> Callable[[Batch], Batch] | None:
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
            self.task_weight = self.task_weight[:num]
        elif len(self.tasks) < num:
            self.tasks += [TaskSelector() for _ in range(num - len(self.tasks))]
            self.task_weight += [
                FloatArg(1.0, min_value=0.0, max_value=1.0) for _ in range(num - len(self.task_weight))
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
    ) -> list[DataLoader | None]:
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
    ) -> list[DataLoader | None]:
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

if __name__ == '__main__':
    class TaskTemp(Task):
        pass

    conf = Tasks.parse_command_line()
    print(conf)
