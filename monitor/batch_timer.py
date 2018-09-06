import math
from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable


class BatchTimer(object):

    def __init__(self):
        self.batches_in_epoch = 1  # will be set next
        self.batch_id = 0

    def init(self, batches_in_epoch: int):
        self.batches_in_epoch = batches_in_epoch

    @property
    def epoch(self):
        return int(self.epoch_progress())

    def epoch_progress(self):
        return self.batch_id / self.batches_in_epoch

    def tick(self):
        self.batch_id += 1


timer = BatchTimer()


class Schedule(ABC):
    """
    Schedule the next update program.
    """
    
    def __init__(self):
        self.last_batch_update = -1

    @abstractmethod
    def next_batch_update(self, batches_in_epoch):
        """
        :param batches_in_epoch: number of batches in one epoch
        :return: the next batch when update is needed
        """
        return 0

    def __call__(self, func: Callable):
        @wraps(func)
        def wrapped(*args, **kwargs):
            if timer.batch_id >= self.next_batch_update(batches_in_epoch=timer.batches_in_epoch):
                self.last_batch_update = timer.batch_id
                func(*args, **kwargs)
        return wrapped


class ScheduleStep(Schedule):
    def __init__(self, epoch_step: int = 1, batch_step: int = 0):
        super().__init__()
        self.epoch_step = epoch_step
        self.batch_step = batch_step

    def next_batch_update(self, batches_in_epoch):
        return batches_in_epoch * self.epoch_step + self.batch_step


class ScheduleExp(Schedule):
    """
    Schedule updates at batches that are powers of two: 1, 2, 4, 8, 16, ...
    Handy for the first epoch.
    """

    def next_batch_update(self, batches_in_epoch):
        if self.last_batch_update > 0:
            next_power = math.floor(math.log2(self.last_batch_update)) + 1
        else:
            next_power = 0
        return 2 ** next_power
