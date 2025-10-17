from typing import Callable
from collections.abc import Iterable

from hyperargs import Conf, IntArg, OptionArg, StrArg, BoolArg, FloatArg, monitor_on, add_dependency
from lmfuser_data import DataLoader, Scanner
from lmfuser_data.interfaces import Index, Row, Batch


class IndexConf(Conf):
    epoch = IntArg(0, min_value=0)
    part = IntArg(0, min_value=0)
    row = IntArg(0, min_value=0)

    def to_index(self) -> Index:
        return Index(self.epoch.value(), self.part.value(), self.row.value()) # type: ignore


@add_dependency('num_workers', 'worker_indexes')
@add_dependency('num_path', 'worker_indexes')
@add_dependency('num_path', 'path_list')
@add_dependency('num_path', 'path_weight')
class DataLoaderConf(Conf):
    batch_size = IntArg(128, min_value=1)
    num_path = IntArg(1, min_value=1)
    path_list = [StrArg('Enther the path to the data file.')]
    path_weight = [FloatArg(1.0, min_value=0.0, max_value=1.0)]
    scanner_type = OptionArg(default='C4Scanner', options=['C4Scanner'])
    seed = IntArg(42)
    shuffle = BoolArg(default=True)
    prefetch_factor = IntArg(2, min_value=0)
    ignore_error = BoolArg(default=True)
    qps = FloatArg(None, min_value=0.1, allow_none=True)
    instruct_timeout = FloatArg(None, min_value=0.1, allow_none=True)
    worker_timeout = FloatArg(None, min_value=0.1, allow_none=True)
    num_workers = IntArg(1, min_value=1)
    num_ranks = IntArg(1, min_value=1)
    rank_idx = IntArg(0, min_value=0)
    worker_indexes = [[IndexConf()]]

    @monitor_on('num_path')
    def set_path_weight(self) -> None:
        num = self.num_path.value()
        assert num is not None
        if len(self.path_weight) > num:
            self.path_weight = self.path_weight[:num]
        elif len(self.path_weight) < num:
            self.path_weight += [
                FloatArg(1.0, min_value=0.0, max_value=1.0) for _ in range(num - len(self.path_weight))
            ]

        if len(self.path_list) > num:
            self.path_list = self.path_list[:num]
        elif len(self.path_list) < num:
            self.path_list += [
                StrArg('Enther the path to the data file.') for _ in range(num - len(self.path_list))
            ]

    @monitor_on('num_workers')
    @monitor_on('num_path')
    def set_worker_indexes(self) -> None:
        num_path = self.num_path.value()
        num_worker_per_path = self.num_workers.value()
        assert num_path is not None
        assert num_worker_per_path is not None
        if len(self.worker_indexes) > num_path:
            self.worker_indexes = self.worker_indexes[:num_path]
        elif len(self.worker_indexes) < num_path:
            self.worker_indexes += [
                [IndexConf()] for _ in range(num_path - len(self.worker_indexes))
            ]

        for worker_indexes in self.worker_indexes:
            if len(worker_indexes) > num_worker_per_path:
                worker_indexes = worker_indexes[:num_worker_per_path]
            elif len(worker_indexes) < num_worker_per_path:
                worker_indexes += [
                    IndexConf() for _ in range(num_worker_per_path - len(worker_indexes))
                ]

    def init_dataloader(
        self, 
        row_map_fn: Callable[[Row], Row] | None = None,
        row_flow_fn: Callable[[Iterable[Row]], Iterable[Row]] | None = None,
        batch_map_fn: Callable[[Batch], Batch] | None = None
    ) -> DataLoader:
        bs = self.batch_size.value()
        assert bs is not None
        
        return DataLoader(
            batch_size=bs if bs is not None else 128,
            path_list=[str(path.value()) for path in self.path_list],
            scanner_type=Scanner.get_subclass(str(self.scanner_type.value())),
            seed=self.seed.value(), # type: ignore
            shuffle=self.shuffle.value(), # type: ignore
            pre_fetch_factor=self.prefetch_factor.value(), # type: ignore
            indexes=[[idx.to_index() for idx in idx_seq] for idx_seq in self.worker_indexes],
            infinite=False,
            map_fn=row_map_fn,
            flow_fn=row_flow_fn,
            ignore_error=self.ignore_error.value(), # type: ignore
            qps=self.qps.value(),
            instruct_timeout=self.instruct_timeout.value(),
            worker_timeout=self.worker_timeout.value(),
            restart_cnt=None,
            num_workers=self.num_workers.value(), # type: ignore
            num_ranks=self.num_ranks.value(), # type: ignore
            rank_idx=self.rank_idx.value(), # type: ignore
            batch_map_fn=batch_map_fn,
            distributor_weights=[weight.value() for weight in self.path_weight] # type: ignore
        )

if __name__ == '__main__':
    conf = DataLoaderConf.parse_command_line()
    conf.init_dataloader()
    print(conf)
