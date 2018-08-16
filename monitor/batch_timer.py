import warnings
from typing import Callable


class BatchTimer(object):

    def __init__(self):
        self.batches_in_epoch = 1  # will be set next
        self.batch_id = 0
        self.schedulers = []

    def init(self, batches_in_epoch: int):
        self.batches_in_epoch = batches_in_epoch
        for scheduler in self.schedulers:
            scheduler.init(batches_in_epoch)

    @property
    def epoch(self):
        return int(self.epoch_progress())

    def epoch_progress(self):
        return self.batch_id / self.batches_in_epoch

    def tick(self):
        self.batch_id += 1


timer = BatchTimer()


class Schedule(object):
    def __init__(self, epoch_update: int = 1, batch_update: int = 0):
        self.epoch_update = epoch_update
        self.batch_update = batch_update
        self.batch_step = 1
        self.next_batch = self.batch_step
        self.initialized = False
        timer.schedulers.append(self)

    def init(self, batches_in_epoch: int):
        self.batch_step = batches_in_epoch * self.epoch_update + self.batch_update
        self.next_batch = self.batch_step
        self.initialized = True

    def need_update(self, current_batch) -> bool:
        if current_batch >= self.next_batch:
            self.next_batch += self.batch_step
            return True
        return False

    def __call__(self, func: Callable):
        def wrapped(*args, **kwargs):
            if not self.initialized:
                warnings.warn("Timer is not initialized!")
            if self.need_update(timer.batch_id):
                func(*args, **kwargs)
        return wrapped
