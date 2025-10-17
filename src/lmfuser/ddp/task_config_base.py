from __future__ import annotations
from typing import Generic

from hyperargs import Conf, BoolArg, monitor_on, add_dependency

from ..dataloader_config import DataLoaderConf
from .task_base import TaskBase, M


@add_dependency('is_trainable', 'train_dataset_config')
@add_dependency('is_evaluatable', 'eval_dataset_config')
class TaskConfigBase(Conf, Generic[M]):

    is_trainable = BoolArg(True)
    is_evaluatable = BoolArg(True)
    train_dataset_config = [DataLoaderConf()]
    eval_dataset_config = [DataLoaderConf()]

    def init_task(self) -> TaskBase[M]:
        raise NotImplementedError('please implement this method!')

    @monitor_on('is_trainable')
    def set_train_config(self) -> None:
        if self.is_trainable.value() and len(self.train_dataset_config) == 0:
            self.train_dataset_config.append(DataLoaderConf())
        elif not self.is_trainable.value():
            self.train_dataset_config = []

    @monitor_on('is_evaluatable')
    def set_eval_config(self) -> None:
        if self.is_evaluatable.value() and len(self.eval_dataset_config) == 0:
            self.eval_dataset_config.append(DataLoaderConf())
        elif not self.is_evaluatable.value():
            self.eval_dataset_config = []

if __name__ == '__main__':
    task_config = TaskConfigBase.parse_command_line()
    print(task_config)
