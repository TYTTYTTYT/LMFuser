from typing import Any, TypeVar, Generic, Iterable, List, Callable
import logging
import abc

from torch import nn
from lmfuser_data.interfaces import Batch, Row

from hyperargs import Conf

logger = logging.getLogger(__name__)

M = TypeVar('M', bound=nn.Module)
T = TypeVar('T', bound=Conf)


class TaskBase(abc.ABC, Generic[M]):
    def is_overridden(self, method_name: str) -> bool:
        """
        Check if the method `method_name` is overridden in this instance's class
        compared to the Parent class.
        """
        cls = self.__class__
        if not hasattr(TaskBase, method_name):
            return False  # No such method in parent

        child_method = getattr(cls, method_name, None)
        parent_method = getattr(TaskBase, method_name, None)

        if not callable(child_method) or not callable(parent_method):
            return False

        try:
            return child_method.__func__ is not parent_method.__func__ # type: ignore
        except AttributeError:
            return child_method is not parent_method

    @abc.abstractmethod
    def train_step(
        self, model: M,
        batch: Batch,
        step: int,
        device: Any,
        **kwargs: Any
    ) -> Batch:
        raise NotImplementedError('Please implement this method in child class')

    @abc.abstractmethod
    def eval_step(
        self,
        model: M,
        batch: Batch,
        step: int,
        device: Any,
        **kwargs: Any
    ) -> dict[str, List[Any]]:
        raise NotImplementedError('Please implement this method in child class')

    @abc.abstractmethod
    def next_train_batch(self, *args, **kwargs) -> Batch:
        raise NotImplementedError('Please implement this method in child class')

    @abc.abstractmethod
    def get_eval_dataloader(self, *args, **kwargs) -> Iterable[Batch]:
        raise NotImplementedError('Please implement this method in child class')

    @abc.abstractmethod
    def cal_dev_metric(self, scores: dict[str, List[Any]]) -> dict[str, Any]:
        raise NotImplementedError('Please implement this method in child class')

    def get_row_processor(self) -> Callable[[Row], Row]:
        return lambda x: x

    def get_batch_processor(self) -> Callable[[Batch], Batch]:
        return lambda x: x

    def get_pipe_processor(self) -> Callable[[Iterable[Row]], Iterable[Row]]:
        return lambda x: x
