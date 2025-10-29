from os import PathLike

from torch import nn
from lmfuser_data.interfaces import SubclassTracer
from hyperargs import Conf, StrArg, OptionArg


class ModelLoader(SubclassTracer):
    def __init__(self, model_path: str | PathLike):
        self.model_path = model_path

    def load_model(self) -> nn.Module:
        raise NotImplementedError(f'Not implemented load_model for {self.__class__.__name__}')

    @classmethod
    def save_model(cls, model: nn.Module, directory: str | PathLike) -> None:
        raise NotImplementedError(f'Not implemented save_model for {model.__class__.__name__}')

def find_model_loader_names() -> list[str]:
    return list(ModelLoader.all_subclass_names()) + ['ModelLoader']


class ModelLoaderConf(Conf):
    model_path = StrArg('please set model path here!')
    model_type = OptionArg(default='ModelLoader', option_fn=find_model_loader_names)

    def get_model_loader(self) -> ModelLoader:
        name = self.model_type.value()
        path = self.model_path.value()
        assert name is not None and path is not None
        if name == 'ModelLoader':
            return ModelLoader(path)
        return ModelLoader.all_subclass_map()[name](path)
