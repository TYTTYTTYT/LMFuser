from typing import TypeVar, Any, Generic
import abc
from logging import Logger, getLogger

import wandb
from wandb.wandb_run import Run

from .train_base_config import TrainConfigBase
from ..utils import get_global_rank, gather_object

T = TypeVar('T', bound=TrainConfigBase)


class TrainBase(abc.ABC, Generic[T]):

    def __init__(self, config: T, *args, **kwargs) -> None:
        self.config = config
        self._step = 1
        self._epoch = 1

    @abc.abstractmethod
    def train(self, **kwargs: Any) -> None:
        raise NotImplementedError(
            'Please implement this method in child classes'
        )

    @abc.abstractmethod
    def eval(self, **kwargs: Any) -> None:
        raise NotImplementedError(
            'Please implement this method in child classes'
        )

    @property
    def _wandb(self) -> Run:
        if getattr(self, '_run', None) is None:
            wandb.init(
                project=self.config.project_name.value(),  # type: ignore
                name=self.config.run_name.value(),  # type: ignore
                config=self.config.to_dict()
            ) if get_global_rank() == 0 else ...
            self._run = True
        return self._run # type: ignore

    @property
    def logger(self) -> Logger:
        if getattr(self, '_logger', None) is None:
            self._logger = getLogger(self.__class__.__name__)
        return self._logger

    def log(self, data: dict[str, Any]) -> None:
        self._wandb
        if get_global_rank() != 0:
            return
        self.logger.info(f'step:{self.step}\t{data}')
        wandb.log(data)

    def step_log(self, data: dict[str, Any]) -> None:
        self._wandb
        if get_global_rank() != 0:
            return
        self.logger.critical(f'step:{self.step}\t{data}')
        wandb.log(data, step=self.step)

    @property
    def step(self) -> int:
        step = getattr(self, '_step', None)
        if step is None:
            self._step = 1
        return self._step

    @property
    def epoch(self) -> int:
        epoch = getattr(self, '_epoch', None)
        if epoch is None:
            self._epoch = 1
        epochs = gather_object(self._epoch)
        epoch = min(epochs)
        return epoch
