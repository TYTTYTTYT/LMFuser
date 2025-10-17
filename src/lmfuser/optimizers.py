from typing import Iterable

import torch
from torch.nn.parameter import Parameter as Parameter
from torch.optim import (
    Optimizer,
    Adam,
    AdamW,
    SGD,
    Adadelta,
    Adagrad
)
from hyperargs import Conf, OptionArg, FloatArg, BoolArg, add_dependency, monitor_on


class OptimizerConfigBase(Conf):

    def init_optimzier(
        self,
        params: Iterable[Parameter],
    ) -> Optimizer:
        raise NotImplementedError(
            'Please Implement this method in child classes!'
        )


class AdamWConfig(OptimizerConfigBase):

    lr = FloatArg(5e-5, min_value=0.0)
    beta1 = FloatArg(0.9, min_value=0.0, max_value=1.0)
    beta2 = FloatArg(0.999, min_value=0.0, max_value=1.0)
    eps = FloatArg(1e-8, min_value=0.0, max_value=1.0)
    weight_decay = FloatArg(0.01, min_value=0.0, max_value=1.0)
    amsgrad = BoolArg(False)

    def init_optimzier(
        self, 
        params: Iterable[Parameter],
    ) -> Optimizer:
        return AdamW(
            params=params,
            lr=self.lr.value(), # type: ignore
            betas=(self.beta1.value(), self.beta2.value()), # type: ignore
            eps=self.eps.value(), # type: ignore
            weight_decay=self.weight_decay.value(), # type: ignore
            amsgrad=self.amsgrad.value() # type: ignore
        )


class AdamConfig(OptimizerConfigBase):

    lr = FloatArg(5e-5, min_value=0.0)
    beta1 = FloatArg(0.9, min_value=0.0, max_value=1.0)
    beta2 = FloatArg(0.999, min_value=0.0, max_value=1.0)
    eps = FloatArg(1e-8, min_value=0.0, max_value=1.0)
    weight_decay = FloatArg(0.01, min_value=0.0, max_value=1.0)
    amsgrad = BoolArg(False)

    def init_optimzier(
        self, 
        params: Iterable[Parameter]
    ) -> Optimizer:
        return Adam(
            params=params,
            lr=self.lr.value(), # type: ignore
            betas=(self.beta1.value(), self.beta2.value()), # type: ignore
            eps=self.eps.value(), # type: ignore
            weight_decay=self.weight_decay.value(), # type: ignore
            amsgrad=self.amsgrad.value() # type: ignore
        )


class SGDConfig(OptimizerConfigBase):

    lr = FloatArg(5e-5, min_value=0.0)
    momentum = FloatArg(0.90, min_value=0.0, max_value=1.0)
    dampening = FloatArg(0.0, min_value=0.0, max_value=1.0)
    weight_decay = FloatArg(0.0, min_value=0.0, max_value=1.0)
    nesterov = BoolArg(False)

    def init_optimzier(
        self, 
        params: Iterable[Parameter],
    ) -> Optimizer:
        return SGD(
            params=params,
            lr=self.lr.value(), # type: ignore
            momentum=self.momentum.value(), # type: ignore
            weight_decay=self.weight_decay.value(), # type: ignore
            nesterov=self.nesterov.value() # type: ignore
        )


class AdadeltaConfig(OptimizerConfigBase):

    lr = FloatArg(1.0, min_value=0.0)
    rho = FloatArg(0.9, min_value=0.0, max_value=1.0)
    eps = FloatArg(1e-6, min_value=0.0, max_value=1.0)
    weight_decay = FloatArg(0.0, min_value=0.0, max_value=1.0)

    def init_optimzier(
        self, 
        params: Iterable[Parameter]
    ) -> Optimizer:
        return Adadelta(
            params=params,
            lr=self.lr.value(), # type: ignore
            rho=self.rho.value(), # type: ignore
            eps=self.eps.value(), # type: ignore
            weight_decay=self.weight_decay.value() # type: ignore
        )


class AdagradConfig(OptimizerConfigBase):

    lr = FloatArg(0.01, min_value=0.0)
    lr_decay = FloatArg(0.0, min_value=0.0, max_value=1.0)
    weight_decay = FloatArg(0.0, min_value=0.0, max_value=1.0)
    initial_accumulator_value = FloatArg(0.0, min_value=0.0, max_value=1.0)
    eps = FloatArg(1e-10, min_value=0.0, max_value=1.0)

    def init_optimzier(
        self, 
        params: Iterable[Parameter]
    ) -> Optimizer:
        return Adagrad(
            params=params,
            lr=self.lr.value(), # type: ignore
            lr_decay=self.lr_decay.value(), # type: ignore
            weight_decay=self.weight_decay.value(), # type: ignore
            initial_accumulator_value=self.initial_accumulator_value.value(), # type: ignore
            eps=self.eps.value() # type: ignore
        )


@add_dependency('optimizer_type', 'optimizer')
class OptimizerConfig(Conf):
    optimizer_type = OptionArg('AdamW', options=['AdamW', 'Adam', 'SGD', 'Adadelta', 'Adagrad'])
    optimizer: OptimizerConfigBase = AdamWConfig()

    @monitor_on('optimizer_type')
    def set_optimizer(self) -> None:
        if self.optimizer_type.value() == 'AdamW' and not isinstance(self.optimizer, AdamWConfig):
            self.optimizer = AdamWConfig()
        elif self.optimizer_type.value() == 'Adam' and not isinstance(self.optimizer, AdamConfig):
            self.optimizer = AdamConfig()
        elif self.optimizer_type.value() == 'SGD' and not isinstance(self.optimizer, SGDConfig):
            self.optimizer = SGDConfig()
        elif self.optimizer_type.value() == 'Adadelta' and not isinstance(self.optimizer, AdadeltaConfig):
            self.optimizer = AdadeltaConfig()
        elif self.optimizer_type.value() == 'Adagrad' and not isinstance(self.optimizer, AdagradConfig):
            self.optimizer = AdagradConfig()

    def init_optimzier(
        self,
        params: Iterable[Parameter]
    ) -> Optimizer:
        return self.optimizer.init_optimzier(params)

if __name__ == '__main__':
    conf = OptimizerConfig.parse_command_line()
    print(conf)
