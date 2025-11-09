from typing import Any, Generic, TypeVar
from abc import ABC, abstractmethod
import os

from hyperargs import Conf, StrArg
from lmfuser_data.interfaces import SubclassTracer


class RunerConf(Conf, SubclassTracer):

    project_name = StrArg('please set a project name')
    run_name = StrArg('please set the name of this run')

ConfType = TypeVar('ConfType', bound=RunerConf)


class Runner(ABC, Generic[ConfType]):
    def __init__(self, config: ConfType, *args, **kwargs) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError('train method not implemented')

    @abstractmethod
    def eval(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError('eval method not implemented')

    @abstractmethod
    def test(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError('test method not implemented')

    @abstractmethod
    def produce(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError('produce method not implemented')

    @abstractmethod
    def save(self, directory: str | os.PathLike, *args, **kwargs) -> None:
        raise NotImplementedError('produce method not implemented')

    @abstractmethod
    def load(self, directory: str | os.PathLike, *args, **kwargs) -> None:
        raise NotImplementedError('load method not implemented')
