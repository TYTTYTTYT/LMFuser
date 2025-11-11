from __future__ import annotations
from typing import Any, Iterator, TypeVar, Generic, Union, Hashable, List, Iterable
from contextlib import ExitStack
from collections import defaultdict
import logging
import os
import random
from pathlib import Path
import json
from logging import Logger, getLogger

import torch
from torch import nn
import torch.distributed
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from torch.amp.grad_scaler import GradScaler
    OLD_GRADSCALER = False
except ImportError:
    from torch.cuda.amp.grad_scaler import GradScaler
    OLD_GRADSCALER = True
from torch.amp.autocast_mode import autocast
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim import Optimizer
from torch import Tensor
from torch.optim.lr_scheduler import LRScheduler
from tqdm import tqdm
from hyperargs import Conf, StrArg, IntArg, FloatArg, OptionArg, BoolArg
from lmfuser_data.interfaces import Batch
import wandb
from wandb.wandb_run import Run

from ..task import Task, Tasks
from ..utils import (
    get_global_rank,
    get_local_rank,
    get_world_size,
    get_default_device,
    dist_init,
    dist_avg,
    batch_all_gather,
    gather_object,
    cal_acc_num,
    get_default_device_type
)

from ..optimizers import OptimizerConfig
from ..schedulers import LRSchedulerConfig
from ..model_loader import ModelLoader, ModelLoaderConf
from .runner import RunerConf, Runner

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=Conf)


class Wrapper(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.module = model.to(get_default_device())
        self.forward = model.forward

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except:
            return self.module.__getattr__(name)


class DDPWraper(nn.Module):

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = DDP(
            model.to(get_default_device()), find_unused_parameters=True
        ) if get_world_size() > 1 else Wrapper(model)
        self.forward = self.model.forward

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except:
            return self.model.module.__getattr__(name)


class DDPRunnerConfig(RunerConf):

    checkpoint_directory = StrArg('please set checkpoint directory here!')
    model_loader_conf = ModelLoaderConf()

    stop_by = OptionArg('step', options=['step', 'epoch'])
    total_step = IntArg(100, min_value=1)
    total_epoch = IntArg(10, min_value=1)
    eval_step_freq = IntArg(100, min_value=1)
    save_step_freq = IntArg(100, min_value=1)

    batch_size = IntArg(32, min_value=1)
    sub_batch_size = IntArg(32, min_value=1)

    grad_norm_clip = FloatArg(None, min_value=0.0, allow_none=True)

    task_conf = Tasks()

    optimizer: OptimizerConfig = OptimizerConfig()
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()

    dp_type = OptionArg(default='ddp', options=['ddp'])
    model_precision = OptionArg(options=['fp32', 'fp16'], default='fp32')
    use_amp = BoolArg(default=False)
    amp_precision = OptionArg(options=['fp16', 'bf16'], default='fp16')
    seed = IntArg(42)

    ignore_data_error = BoolArg(default=False)
    data_row_qps = FloatArg(None, min_value=0.0, allow_none=True)
    instruct_timeout = FloatArg(30.0, min_value=0.0)
    worker_timeout = FloatArg(30.0, min_value=0.0)
    shuffle_dataset = BoolArg(default=True)
    row_prefetch = IntArg(0, min_value=0)
    num_row_workers = IntArg(1, min_value=1)

    resume_training = BoolArg(default=False)
    resume_path = StrArg(default=None, allow_none=True)

    @property
    def _default_precision(self) -> torch.dtype:
        if self.model_precision == 'fp32':
            return torch.float32
        elif self.model_precision == 'fp16':
            return torch.float16
        else:
            raise ValueError(self.model_precision)

    @property
    def _num_acc_steps(self) -> int:
        bs = self.batch_size.value()
        sbs = self.sub_batch_size.value()
        assert bs is not None and sbs is not None

        if bs % (sbs * get_world_size()) != 0:
            raise ValueError(
                f'batch_size ({bs}) must be divisible by sub_batch_size * world_size ({sbs * get_world_size()})'
            )
        return bs // (sbs * get_world_size())


class DDPRunner(Runner[DDPRunnerConfig]):

    def __init__(self, config: DDPRunnerConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        if get_world_size() > 1:
            dist_init()

        self.tasks = [task.conf for task in config.task_conf.tasks]
        self.step = 1
        self.pre_epoch = 0
        self.model_loader = config.model_loader_conf.get_model_loader()

        if config.resume_training.value():
            resume_path = config.resume_path.value()
            assert resume_path is not None, 'resume_path is None'
            self.load(resume_path)
        self.config.seed = self.config.seed.parse(hash(f'original_seed_{self.config.seed.value()}|step_{self.step}'))

        self.train_data_loaders = config.task_conf.get_train_dataloaders(
            batch_size=config.sub_batch_size.value(), # type: ignore
            seed=config.seed.value(), # type: ignore
            shuffle=config.shuffle_dataset.value(), # type: ignore
            prefetch_factor=config.row_prefetch.value(), # type: ignore
            num_workers=config.num_row_workers.value(), # type: ignore
            ignore_error=config.ignore_data_error.value(), # type: ignore
            qps=config.data_row_qps.value(),
            instruct_timeout=config.instruct_timeout.value(), # type: ignore
            worker_timeout=config.worker_timeout.value(), # type: ignore
            world_size=get_world_size(),
            rank=get_global_rank(),
        )

        self.eval_data_loaders = config.task_conf.get_eval_dataloaders(
            batch_size=config.sub_batch_size.value(), # type: ignore
            seed=config.seed.value(), # type: ignore
            shuffle=config.shuffle_dataset.value(), # type: ignore
            prefetch_factor=config.row_prefetch.value(), # type: ignore
            num_workers=config.num_row_workers.value(), # type: ignore
            ignore_error=config.ignore_data_error.value(), # type: ignore
            qps=config.data_row_qps.value(),
            instruct_timeout=config.instruct_timeout.value(), # type: ignore
            worker_timeout=config.worker_timeout.value(), # type: ignore
            world_size=get_world_size(),
            rank=get_global_rank(),
        )

        self.test_data_loaders = config.task_conf.get_test_dataloaders(
            batch_size=config.sub_batch_size.value(), # type: ignore
            seed=config.seed.value(), # type: ignore
            shuffle=config.shuffle_dataset.value(), # type: ignore
            prefetch_factor=config.row_prefetch.value(), # type: ignore
            num_workers=config.num_row_workers.value(), # type: ignore
            ignore_error=config.ignore_data_error.value(), # type: ignore
            qps=config.data_row_qps.value(),
            instruct_timeout=config.instruct_timeout.value(), # type: ignore
            worker_timeout=config.worker_timeout.value(), # type: ignore
            world_size=get_world_size(),
            rank=get_global_rank(),
        )

        self.train_task_idxs: list[int] = []
        for idx, loader in enumerate(self.train_data_loaders):
            if loader is not None:
                self.train_task_idxs.append(idx)

        self.eval_task_idxs: list[int] = []
        for idx, loader in enumerate(self.eval_data_loaders):
            if loader is not None:
                self.eval_task_idxs.append(idx)

        self.test_task_idxs: list[int] = []
        for idx, loader in enumerate(self.test_data_loaders):
            if loader is not None:
                self.test_task_idxs.append(idx)

        assert len(self.train_task_idxs) + len(self.eval_task_idxs) > 0, 'At least one train or eval task must be provided.'

        self.task_weights: list[float] = [w.value() for w in config.task_conf.task_weights] # type: ignore
        self.train_task_weights = [self.task_weights[idx] for idx in self.train_task_idxs]
        self.eval_task_weights = [self.task_weights[idx] for idx in self.eval_task_idxs]
        self.task_rand_g = random.Random(config.seed.value())

        self.train_iters = [iter(loader) if loader is not None else None for loader in self.train_data_loaders]
        self.eval_iters = [iter(loader) if loader is not None else None for loader in self.eval_data_loaders]

        self._all_eval_results: dict[str, list[dict[str, Any]]] = {}
        self._test_results: dict[str, dict[str, Any]] = {}

    def sample_train_task_id(self) -> int:
        '''
        by default, randomly select a task from the task list.
        You can modify this fuction to control the task scheduler.
        Be sure that all ranks are selecting the same task at the same time.
        '''
        return self.task_rand_g.choices(
            list(range(len(self.train_task_idxs))), weights=self.train_task_weights, k=1
        )[0]

    def sample_eval_task_id(self) -> int:
        '''
        by default, randomly select a task from the task list.
        You can modify this fuction to control the task scheduler.
        Be sure that all ranks are selecting the same task at the same time.
        '''
        return self.task_rand_g.choices(
            list(range(len(self.eval_task_idxs))), weights=self.eval_task_weights, k=1
        )[0]

    def load_model(self, **kwargs: Any) -> nn.Module:
        return self.model_loader.load_model()

    def save(self, model: nn.Module, directory: str, step: int, **kwargs: Any) -> None:
        path = Path(directory) / str(step)
        os.makedirs(path, exist_ok=True)
        self.model_loader.save_model(model, path)

        optimizer_path = path / 'optimizer.pt'
        torch.save(self.optimizer.state_dict(), optimizer_path)

        scheduler_path = path / 'scheduler.pt'
        torch.save(self.scheduler.state_dict(), scheduler_path)

        runner_path = path / 'runner.json'
        with open(runner_path, 'w') as f:
            f.write(json.dumps({
                'step': step + 1,
                'epoch': self.epoch,
                'config': self.config.to_dict(),
            }, indent=4))

    @property
    def model(self) -> DDPWraper:
        model = getattr(self, '_model', None)
        if model is None:
            model = self.load_model()
            if self.config.model_precision.value() == 'fp16':
                logger.critical(f'casting model to fp16')
                model = model.half()
            self._model = DDPWraper(model)
        assert self._model is not None
        return self._model

    def _should_stop(self) -> bool:
        stop_metric = self.config.stop_by.value()
        assert stop_metric in ('step', 'epoch')

        if stop_metric == 'step':
            total_step = self.config.total_step.value()
            assert total_step is not None
            return self.step > total_step
        elif stop_metric == 'epoch':
            total_epoch = self.config.total_epoch.value()
            assert total_epoch is not None
            return self.epoch >= total_epoch
        else:
            raise ValueError(f'stop_metric must be either "epoch" or "step", got "{stop_metric}" instead.')

    def _batch_to_device(self, batch: Batch) -> Batch:
        for key, v in batch.items():
            if isinstance(v, Tensor):
                if torch.is_floating_point(v):
                    precision = self.config.model_precision.value()
                    if precision == 'fp32':
                        v = v.to(torch.float32) if v.dtype != torch.float32 else v
                    elif precision == 'fp16':
                        v = v.to(torch.float16) if v.dtype != torch.float16 else v
                    else:
                        raise ValueError(f'Unknown model precision "{precision}"')
                assert isinstance(batch, dict)
                batch[key] = v.to(get_default_device()) if v.get_device() != get_default_device() else v
        return batch

    def _prepare_train(
        self, 
        optimizer: Union[Optimizer, None] = None,
        scheduler: Union[LRScheduler, None] = None,
        scaler: Union[GradScaler, None] = None,
    ) -> None:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        if optimizer is None:
            self.optimizer = self.config.optimizer.init_optimzier(
                self.model.parameters()
            )
        else:
            self.optimizer = optimizer
        if hasattr(self, '_optimizer_states'):
            self.optimizer.load_state_dict(self._optimizer_states)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(get_default_device(), non_blocking=True)

        if scheduler is None:
            self.scheduler = self.config.lr_scheduler.init_lr_scheduler(
                self.optimizer
            )
        else:
            self.scheduler = scheduler
        if hasattr(self, '_scheduler_states'):
            self.scheduler.load_state_dict(self._scheduler_states)

        if scaler is None:
            if self.config.use_amp.value() and self.config.amp_precision.value() == 'fp16':
                enable_scaler = True
            else:
                enable_scaler = False
            if self.config.model_precision.value() == 'fp16':
                enable_scaler = False

            if OLD_GRADSCALER:
                self.scaler = GradScaler(enabled=enable_scaler)
            else:
                self.scaler = GradScaler(device=get_default_device_type(), enabled=enable_scaler) # type: ignore
        else:
            self.scaler = scaler

    def _next_train_batch(self, task_idx: int) -> Batch:
        if self.train_data_loaders[task_idx] is None:
            raise ValueError(f'No train dataloader for task {task_idx}')

        it = self.train_iters[task_idx]
        assert it is not None
        try:
            return next(it)
        except StopIteration:
            it = iter(self.train_data_loaders[task_idx]) # type: ignore
            self.train_iters[task_idx] = it
            return next(it)

    @property
    def _wandb(self) -> Run:
        if getattr(self, '_run', None) is None:
            wandb.init(
                project=self.config.project_name.value(),
                name=self.config.run_name.value(),
                config=self.config.to_dict()
            ) if get_global_rank() == 0 else ...
            self._run = True
        return self._run # type: ignore

    @property
    def logger(self) -> Logger:
        if getattr(self, '_logger', None) is None:
            self._logger = getLogger(self.__class__.__name__)
        return self._logger

    @property
    def epoch(self) -> int:
        epochs = [loader.epoch for loader in self.train_data_loaders if loader is not None]
        return max(epochs) + self.pre_epoch

    def step_log(self, data: dict[str, Any]) -> None:
        self._wandb
        if get_global_rank() != 0:
            return
        self.logger.critical(f'step:{self.step}\t{data}')
        wandb.log(data, step=self.step)

    def _one_train_step(self, **kwargs: Any) -> None:
        # clean the gradients
        self.optimizer.zero_grad()

        # select a task to run in this step
        task_id = self.sample_train_task_id()
        task = self.tasks[task_id]

        # calculate loss
        running_loss: float = 0.0
        batch_datas: defaultdict[Hashable, List[float]] = defaultdict(list)
        for acc_idx in range(self.config._num_acc_steps):
            with ExitStack() as stack:
                # check whether to use amp
                if self.config.use_amp.value():
                    amp_selection = self.config.amp_precision.value()
                    assert amp_selection is not None
                    stack.enter_context(autocast(
                        device_type='cuda',
                        dtype={
                            'fp16': torch.float16,
                            'bf16': torch.bfloat16
                        }[amp_selection],
                    ))

                # compute loss for each sub_batch
                subbatch_result = task.train_step(
                    model=self.model,
                    batch=self._batch_to_device(self._next_train_batch(task_id)),
                    step=self.step, 
                    device=get_local_rank(),
                    acc_step=acc_idx,
                )
                if isinstance(subbatch_result, torch.Tensor):
                    subbatch_result = {'loss': subbatch_result}
                if 'loss' not in subbatch_result:
                    raise KeyError('no loss returned from the batch')
                assert isinstance(subbatch_result, dict)
                loss = subbatch_result['loss']
                assert isinstance(loss, Tensor)
                loss = loss / self.config._num_acc_steps
                running_loss += loss.item()
                for k, v in subbatch_result.items():
                    if k == 'loss':
                        continue
                    if isinstance(v, (float, int)):
                        batch_datas[k].append(float(v))
                    elif isinstance(v, (list)) and len(v) > 0 and isinstance(v[0], (float, int)):
                        batch_datas[k].extend([float(i) for i in v]) # type: ignore

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

        running_loss = dist_avg(running_loss)
        self.step_log({f'{task.__class__.__name__}/train/loss': running_loss})
        self.step_log({'train/epoch': self.epoch})
        for k, v in batch_datas.items():
            try:
                avg = sum(v) / len(v)
            except:
                continue
            self.step_log({f'{task.__class__.__name__}/train/{k}': avg})
        self._pbar_train.set_description(
            f'train loss: {running_loss:.3g}', refresh=True
        )

        grad_norm_clip_val = self.config.grad_norm_clip.value()
        if grad_norm_clip_val is not None:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            norm = clip_grad_norm_(
                parameters=self.model.parameters(), 
                max_norm=grad_norm_clip_val
            ).item()
            norm = dist_avg(norm)
            self.step_log({f'{task.__class__.__name__}/train/grad_norm': norm})

        num_hot_params = 0
        num_freeze_params = 0
        for param in self.model.parameters():
            if param.requires_grad == False:
                param.grad = None
                num_freeze_params += param.numel()
            else:
                num_hot_params += param.numel()
        num_total_params = num_hot_params + num_freeze_params
        if num_total_params == 0:
            raise RuntimeError('The model contains no parameters.')

        self.step_log({
            f'{task.__class__.__name__}/train/num_hot_params': num_hot_params,
            f'{task.__class__.__name__}/train/num_freeze_params': num_freeze_params,
            f'{task.__class__.__name__}/train/num_total_params': num_total_params,
            f'{task.__class__.__name__}/train/hot_ratio': num_hot_params / num_total_params,
        })

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        current_lr = self.scheduler.get_lr()
        if isinstance(current_lr, list):
            current_lr = current_lr[0]
        self.step_log(
            {f'{task.__class__.__name__}/train/learning_rate': current_lr}
        )
        self.scheduler.step()
        torch.distributed.barrier() if get_world_size() > 1 else ...

        save_step_freq = self.config.save_step_freq.value()
        assert save_step_freq is not None
        if all(
            [self.step % save_step_freq == 0, get_global_rank() == 0]
        ):
            logger.info('begin to save the model')
            self._pbar_train.set_description('begin to save the model', True)
            to_save = self.model.model.module
            ckpt_path = self.config.checkpoint_directory.value()
            assert ckpt_path is not None
            self.save(
                to_save,  # type: ignore
                ckpt_path,
                self.step
            )
            logger.info('model saved!')
            self._pbar_train.set_description('model saved!', True)
        torch.distributed.barrier() if get_world_size() > 1 else ...

        eval_step_freq = self.config.eval_step_freq.value()
        assert eval_step_freq is not None
        if self.step % eval_step_freq == 0:
            logger.info('begin to evaluate the model')
            self.eval()
            logger.info('model evaluated!')
        self.step += 1

    def train(self, *args: Any, **kwargs: Any) -> None:
        self._prepare_train()

        self._pbar_train = tqdm(
            total=self.config.total_step.value(),
            position=0,
            initial=self.step - 1,
            dynamic_ncols=True,
            unit='step',
            disable=True if get_global_rank() != 0 else False
        )

        while not self._should_stop():
            self._one_train_step()
            self._pbar_train.update(1)

        self.test()
        self._pbar_train.close()

    def _eval_one_task(self, task_id: int, **kwargs: Any) -> None:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        dataloader = self.eval_data_loaders[task_id]
        task = self.tasks[task_id]

        batch_list: list[dict[str, list[Any]]] = []
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if get_world_size() > 1:
                stack.enter_context(self.model.model.no_sync())
            for batch in tqdm(
                dataloader,
                dynamic_ncols=True,
                desc=f'evaluating task {task.__class__.__name__}',
                position=3,
                disable=True if get_global_rank() != 0 else False,
                leave=False,
                unit='batch'
            ):
                result = task.eval_step(
                    self.model, # type: ignore
                    self._batch_to_device(batch),
                    self.step,
                    get_default_device()
                )
                batch_list.append(result)

            if len(batch_list) == 0:
                raise RuntimeError(f'No eval data found in rank {get_global_rank()} '
                                f'for task {task.__class__.__name__}')
            cat_result = batch_list[0]
            for batch in batch_list[1:]:
                for k, v in batch.items():
                    cat_result[k] += v
        all_result = batch_all_gather(cat_result)

        with torch.no_grad():
            metrics = task.cal_dev_metric(all_result)
        for k, v in metrics.items():
            self.step_log({f'{task.__class__.__name__}/dev/{k}': v})

        if task.__class__.__name__ not in self._all_eval_results:
            self._all_eval_results[task.__class__.__name__] = []
        self._all_eval_results[task.__class__.__name__].append({'step': self.step, 'metrics': metrics})

    def _test_one_task(self, task_id: int, **kwargs: Any) -> None:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        dataloader = self.test_data_loaders[task_id]
        task = self.tasks[task_id]

        ckpt_path = self.config.checkpoint_directory.value()
        assert ckpt_path is not None
        test_model_path = task.set_test_model_path(ckpt_path, self._all_eval_results)
        if test_model_path is not None:
            if isinstance(test_model_path, int):
                test_model_path = Path(ckpt_path) / str(test_model_path)
            
            self.model_loader.model_path = test_model_path
            self._model = None

        batch_list: list[dict[str, list[Any]]] = []
        with ExitStack() as stack:
            stack.enter_context(torch.no_grad())
            if get_world_size() > 1:
                stack.enter_context(self.model.model.no_sync())
            for batch in tqdm(
                dataloader,
                dynamic_ncols=True,
                desc=f'testing task {task.__class__.__name__}',
                position=4,
                disable=True if get_global_rank() != 0 else False,
                leave=False,
                unit='batch'
            ):
                result = task.eval_step(
                    self.model, # type: ignore
                    self._batch_to_device(batch),
                    self.step,
                    get_default_device()
                )
                batch_list.append(result)

            if len(batch_list) == 0:
                raise RuntimeError(f'No test data found in rank {get_global_rank()} '
                                f'for task {task.__class__.__name__}')
            cat_result = batch_list[0]
            for batch in batch_list[1:]:
                for k, v in batch.items():
                    cat_result[k] += v
        all_result = batch_all_gather(cat_result)

        with torch.no_grad():
            metrics = task.cal_dev_metric(all_result)
        for k, v in metrics.items():
            self.step_log({f'{task.__class__.__name__}/test/{k}': v})

        self._test_results[task.__class__.__name__] = {'step': self.step, 'metrics': metrics}

    def eval(self, *args: Any, **kwargs: Any) -> None:
        for task_id in tqdm(
            self.eval_task_idxs, 
            dynamic_ncols=True, 
            desc=f'evaluating {len(self.tasks)} tasks...',
            position=2,
            disable=True if get_global_rank() != 0 else False,
            leave=False,
            unit='task'
        ):
            self._eval_one_task(task_id)

        if get_global_rank() != 0:
            return
        ckpt_path = self.config.checkpoint_directory.value()
        assert ckpt_path is not None
        try:
            eval_results = json.dumps(self._all_eval_results, indent=2)
            ckpt_dir = Path(ckpt_path)
            if not ckpt_dir.exists():
                ckpt_dir.mkdir(parents=True)
            eval_path = ckpt_dir / 'eval_results.json'
            with open(eval_path, 'w') as f:
                f.write(eval_results)
        except:
            logger.warning(f'Failed to save eval results to {ckpt_path}')

    def test(self, *args: Any, **kwargs: Any) -> None:
        for task_id in tqdm(
            self.test_task_idxs,
            dynamic_ncols=True, 
            desc=f'testing {len(self.test_task_idxs)} tasks...',
            position=3,
            disable=True if get_global_rank() != 0 else False,
            leave=False,
            unit='task'
        ):
            self._test_one_task(task_id)

        if get_global_rank() != 0:
            return
        ckpt_path = self.config.checkpoint_directory.value()
        assert ckpt_path is not None
        try:
            test_results = json.dumps(self._test_results, indent=2)
            ckpt_dir = Path(ckpt_path)
            if not ckpt_dir.exists():
                ckpt_dir.mkdir(parents=True)
            test_path = ckpt_dir / 'test_results.json'
            with open(test_path, 'w') as f:
                f.write(test_results)
        except:
            logger.warning(f'Failed to save test results to {ckpt_path}')

    def produce(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError('produce method not implemented')

    def load(self, directory: str | os.PathLike, *args, **kwargs) -> None:
        self.model_loader.model_path = directory

        optimizer_path = Path(directory) / 'optimizer.pt'
        if optimizer_path.exists():
            self._optimizer_states = torch.load(optimizer_path)
        else:
            logger.warning(f'optimizer.pt not found in {directory}, skip loading optimizer')

        scheduler_path = Path(directory) / 'scheduler.pt'
        if scheduler_path.exists():
            self._scheduler_states = torch.load(scheduler_path)
        else:
            logger.warning(f'scheduler.pt not found in {directory}, skip loading scheduler')

        runner_path = Path(directory) / 'runner.json'
        if runner_path.exists():
            with open(runner_path, 'r') as f:
                runner_state = json.load(f)
                self.step = int(runner_state['step'])
                self.pre_epoch = int(runner_state['epoch'])
        else:
            logger.warning(f'runner.json not found in {directory}, skip loading runner state')
