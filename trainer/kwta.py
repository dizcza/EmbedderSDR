from typing import Union, Optional

import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from model import KWinnersTakeAllSoft, KWinnersTakeAll
from trainer.gradient import TrainerGrad
from utils import find_layers


class KWTAScheduler:
    """
    KWinnersTakeAll sparsity scheduler.
    """

    def __init__(self, model: nn.Module, step_size: int, gamma=0.5, min_sparsity=0.05, last_epoch=-1):
        self.kwta_layers = tuple(find_layers(model, layer_class=KWinnersTakeAll))
        self.base_sparsity = tuple(layer.sparsity for layer in self.kwta_layers)
        self.step_size = step_size
        self.gamma = gamma
        self.min_sparsity = min_sparsity
        self.last_epoch = last_epoch

    def get_sparsity(self):
        sparsity_layers = []
        epoch = max(self.last_epoch, 0)
        for base_sparsity in self.base_sparsity:
            sparsity = base_sparsity * self.gamma ** (epoch // self.step_size)
            sparsity = max(sparsity, self.min_sparsity)
            sparsity_layers.append(sparsity)
        return sparsity_layers

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for layer, sparsity in zip(self.kwta_layers, self.get_sparsity()):
            layer.sparsity = sparsity

    def extra_repr(self):
        return f"step_size={self.step_size}, gamma={self.gamma}, min_sparsity={self.min_sparsity}"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"


class TrainerGradKWTA(TrainerGrad):

    watch_modules = TrainerGrad.watch_modules + (KWinnersTakeAll,)

    """
    If the model does not have any KWinnerTakeAll layers, it works just as TrainerGrad.
    """

    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau, None] = None,
                 kwta_scheduler: Optional[KWTAScheduler] = None,
                 **kwargs):
        super().__init__(model=model, criterion=criterion, dataset_name=dataset_name, optimizer=optimizer,
                         scheduler=scheduler, **kwargs)
        self.kwta_scheduler = kwta_scheduler
        self.monitor.register_func(lambda: [layer.hardness.item() for layer in
                                            find_layers(self.model, layer_class=KWinnersTakeAllSoft)], opts=dict(
            xlabel='Epoch',
            ylabel='hardness',
            title='k-winner-take-all hardness parameter'
        ))
        if self.kwta_scheduler is not None:
            self.monitor.register_func(lambda: self.kwta_scheduler.get_sparsity(), opts=dict(
                xlabel='Epoch',
                ylabel='sparsity',
                title='KWinnersTakeAll sparsity',
                ytype='log',
            ))

    def log_trainer(self):
        super().log_trainer()
        self.monitor.log(f"KWTA scheduler: {self.kwta_scheduler}")

    @property
    def env_name(self) -> str:
        env_name = super().env_name
        if not any(find_layers(self.model, layer_class=KWinnersTakeAll)):
            env_name += " no-kwta"
        return env_name

    def _epoch_finished(self, epoch, outputs, labels) -> torch.Tensor:
        loss = super()._epoch_finished(epoch, outputs, labels)
        if self.kwta_scheduler is not None:
            self.kwta_scheduler.step(epoch=epoch)
        return loss
