from typing import Union, Optional

import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from model.kwta import KWinnersTakeAllSoft, KWinnersTakeAll
from trainer.gradient import TrainerGrad
from utils import find_layers


class KWTAScheduler:
    """
    KWinnersTakeAll sparsity scheduler.
    """

    def __init__(self, model: nn.Module, step_size: int, gamma_sparsity=0.5, min_sparsity=0.05,
                 gamma_hardness=2.0, max_hardness=10):
        self.kwta_layers = tuple(find_layers(model, layer_class=KWinnersTakeAll))
        self.step_size = step_size
        self.gamma_sparsity = gamma_sparsity
        self.min_sparsity = min_sparsity
        self.gamma_hardness = gamma_hardness
        self.max_hardness = max_hardness
        self.epoch = 0
        self.last_epoch_update = -1

    def need_update(self):
        return self.epoch >= self.last_epoch_update + self.step_size

    def step(self, epoch=None):
        if epoch is not None:
            self.epoch = epoch
        if self.need_update():
            for layer in self.kwta_layers:
                layer.sparsity = max(layer.sparsity * self.gamma_sparsity, self.min_sparsity)
                if isinstance(layer, KWinnersTakeAllSoft):
                    layer.hardness = min(layer.hardness * self.gamma_hardness, self.max_hardness)
            self.last_epoch_update = self.epoch
        self.epoch += 1

    def extra_repr(self):
        return f"step_size={self.step_size}, Sparsity(gamma={self.gamma_sparsity}, min={self.min_sparsity}), " \
               f"Hardness(gamma={self.gamma_hardness}, max={self.max_hardness})"

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
        if self.kwta_scheduler is not None:
            kwta_layers = tuple(find_layers(model, layer_class=KWinnersTakeAll))
            kwta_layers_soft = tuple(find_layers(model, layer_class=KWinnersTakeAllSoft))
            self.monitor.register_func(lambda: [layer.sparsity for layer in kwta_layers], opts=dict(
                xlabel='Epoch',
                ylabel='sparsity',
                title='KWinnersTakeAll.sparsity',
                ytype='log',
            ))
            self.monitor.register_func(lambda: [layer.hardness for layer in kwta_layers_soft], opts=dict(
                xlabel='Epoch',
                ylabel='hardness',
                title='KWinnersTakeAllSoft.hardness',
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
