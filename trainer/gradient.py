from typing import Union

import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from trainer.trainer import Trainer


class TrainerGrad(Trainer):
    """
    Default gradient descent trainer with full float precision.
    """

    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau, None] = None,
                 **kwargs):
        super().__init__(model, criterion, dataset_name, **kwargs)
        self.optimizer = optimizer
        self.scheduler = scheduler

    def monitor_functions(self):
        super().monitor_functions()

        def learning_rate(viz):
            viz.line_update(y=[group['lr'] for group in self.optimizer.param_groups], opts=dict(
                xlabel='Epoch',
                ylabel='Learning rate',
                title='Learning rate',
                ytype='log',
            ))

        if self.scheduler is not None:
            self.monitor.register_func(learning_rate)

    def log_trainer(self):
        super().log_trainer()
        optimizer_str = f"Optimizer {self.optimizer.__class__.__name__}:"
        for group_id, group in enumerate(self.optimizer.param_groups):
            optimizer_str += f"\n\tgroup {group_id}: lr={group['lr']}, weight_decay={group['weight_decay']}"
        self.monitor.log(optimizer_str)

    def train_batch(self, images, labels):
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        loss.backward()
        self.optimizer.step(closure=None)
        return outputs, loss

    def _epoch_finished(self, epoch, outputs, labels):
        loss = super()._epoch_finished(epoch, outputs, labels)
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(metrics=loss, epoch=epoch)
        elif isinstance(self.scheduler, _LRScheduler):
            self.scheduler.step(epoch=epoch)
        return loss

    def state_dict(self):
        state = super().state_dict()
        state['optimizer'] = self.optimizer.state_dict()
        state['criterion'] = self.criterion.state_dict()
        return state

    def restore(self, checkpoint_path=None, strict=True):
        checkpoint_state = super().restore(checkpoint_path=checkpoint_path, strict=strict)
        try:
            if checkpoint_state is not None:
                self.optimizer.load_state_dict(checkpoint_state['optimizer'])
                self.criterion.load_state_dict(checkpoint_state['criterion'])
        except Exception as exception:
            print("Couldn't restore optimizer: ", exception)
        return checkpoint_state
