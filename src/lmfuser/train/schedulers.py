from functools import partial
import math

from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.optim import Optimizer

from hyperargs import Conf, IntArg, FloatArg, OptionArg, monitor_on, add_dependency

def _get_linear_schedule_with_warmup_lr_lambda(
    current_step: int, 
    *, 
    num_warmup_steps: int, 
    num_training_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(
        0.0, 
        float(num_training_steps - current_step) / float(
            max(1, num_training_steps - num_warmup_steps))
        )

def get_linear_schedule_with_warmup(
    optimizer: Optimizer,
    num_warmup_steps: int, 
    num_training_steps: int, 
    last_epoch=-1
) -> LRScheduler:
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_linear_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def _get_constant_schedule_with_warmup_lr_lambda(
    current_step: int, 
    *, 
    num_warmup_steps: int
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1.0, num_warmup_steps))
    return 1.0

def get_constant_schedule_with_warmup(
    optimizer: Optimizer, 
    num_warmup_steps: int, 
    last_epoch: int = -1
) -> LRScheduler:
    """
    Create a schedule with a constant learning rate preceded by a warmup period during which the learning rate
    increases linearly between 0 and the initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(_get_constant_schedule_with_warmup_lr_lambda, num_warmup_steps=num_warmup_steps)
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def _get_cosine_schedule_with_warmup_lr_lambda(
    current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_cycles: float
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
) -> LRScheduler:
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_lambda = partial(
        _get_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_cycles=num_cycles,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


class LRSchedulerConfigBase(Conf):

    def init_lr_scheduler(self, optimizer: Optimizer, **kwargs) -> LRScheduler:
        raise NotImplementedError(
            'Please Implement this method in child classes!'
        )


class ConstantScheduleWithWarmupConfig(LRSchedulerConfigBase):

    num_warmup_steps= IntArg(0, min_value=0)
    last_epoch= IntArg(-1, min_value=-1)

    def init_lr_scheduler(self, optimizer: Optimizer, **kwargs) -> LRScheduler:
        return get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps.value(), # type: ignore
            last_epoch=self.last_epoch.value() # type: ignore
        )


class LienarScheduleWithWarmup(LRSchedulerConfigBase):

    num_warmup_steps= IntArg(0, min_value=0)
    num_training_steps= IntArg(10000, min_value=1)
    last_epoch= IntArg(-1, min_value=-1)

    def init_lr_scheduler(self, optimizer: Optimizer, **kwargs) -> LRScheduler:
        return get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps.value(), # type: ignore
            num_training_steps=self.num_training_steps.value(), # type: ignore
            last_epoch=self.last_epoch.value() # type: ignore
        )


class CosineScheduleWithWarmu(LRSchedulerConfigBase):
    
    num_warmup_steps= IntArg(0, min_value=0)
    num_training_steps= IntArg(10000, min_value=1)
    num_cycles= FloatArg(0.5, min_value=0.0, max_value=1.0)
    last_epoch= IntArg(-1, min_value=-1)

    def init_lr_scheduler(self, optimizer: Optimizer, **kwargs) -> LRScheduler:
        return get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps.value(), # type: ignore
            num_training_steps=self.num_training_steps.value(), # type: ignore
            num_cycles=self.num_cycles.value(), # type: ignore
            last_epoch=self.last_epoch.value() # type: ignore
        )


@add_dependency('type', 'scheduler')
class LRSchedulerConfig(LRSchedulerConfigBase):

    type = OptionArg(
        options=[
            'constant_schedule_with_warmup',
            'linear_schedule_with_warmup',
            'cosine_schedule_with_warmup'
        ],
        default='constant_schedule_with_warmup',
    )
    scheduler = ConstantScheduleWithWarmupConfig()

    def init_lr_scheduler(self, optimizer: Optimizer, **kwargs) -> LRScheduler:
        return self.scheduler.init_lr_scheduler(optimizer, **kwargs)

    @monitor_on('type')
    def set_scheduler(self) -> None:
        if self.type.value() == 'constant_schedule_with_warmup' and not isinstance(self.scheduler, ConstantScheduleWithWarmupConfig):
            self.scheduler = ConstantScheduleWithWarmupConfig()
        elif self.type.value() == 'linear_schedule_with_warmup' and not isinstance(self.scheduler, LienarScheduleWithWarmup):
            self.scheduler = LienarScheduleWithWarmup()
        elif self.type.value() == 'cosine_schedule_with_warmup' and not isinstance(self.scheduler, CosineScheduleWithWarmu):
            self.scheduler = CosineScheduleWithWarmu()

if __name__ == '__main__':
    conf = LRSchedulerConfig.parse_command_line()
    print(conf)
