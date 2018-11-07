import warnings
from typing import Union, Optional

import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from loss import LossFixedPattern
from models.kwta import KWinnersTakeAllSoft, KWinnersTakeAll, SynapticScaling
from trainer.gradient import TrainerGrad
from trainer.mask import MaskTrainerIndex
from utils.layers import find_layers, find_named_layers
from utils.prepare import prepare_eval


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
        self.last_epoch_update = -1

    def need_update(self, epoch: int):
        return epoch >= self.last_epoch_update + self.step_size

    def step(self, epoch: int):
        if self.need_update(epoch):
            for layer in self.kwta_layers:
                layer.sparsity = max(layer.sparsity * self.gamma_sparsity, self.min_sparsity)
                if isinstance(layer, KWinnersTakeAllSoft):
                    layer.hardness = min(layer.hardness * self.gamma_hardness, self.max_hardness)
            self.last_epoch_update = epoch

    def extra_repr(self):
        return f"step_size={self.step_size}, Sparsity(gamma={self.gamma_sparsity}, min={self.min_sparsity}), " \
               f"Hardness(gamma={self.gamma_hardness}, max={self.max_hardness})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"


class TrainerGradKWTA(TrainerGrad):
    watch_modules = TrainerGrad.watch_modules + (KWinnersTakeAll, SynapticScaling)

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
        if not any(find_layers(self.model, layer_class=KWinnersTakeAll)):
            self.env_name = self.env_name + " (TrainerGrad)"
        self.kwta_scheduler = kwta_scheduler
        if self.kwta_scheduler is not None and isinstance(self.criterion, LossFixedPattern):
            warnings.warn(f"{self.kwta_scheduler.__class__.__name__} is not recommended to use with "
                          f"{self.criterion.__class__.__name__}. Make sure kWTA sparsity does not "
                          f"change during the training.")
        self.mask_trainer_kwta = MaskTrainerIndex(image_shape=self.mask_trainer.image_shape)

    def monitor_functions(self):
        super().monitor_functions()
        kwta_named_layers = tuple(find_named_layers(self.model, layer_class=KWinnersTakeAll))

        if self.kwta_scheduler is not None:
            kwta_named_layers_soft = tuple(find_named_layers(self.model, layer_class=KWinnersTakeAllSoft))

            def sparsity(viz):
                layers_sparsity = []
                names = []
                for name, layer in kwta_named_layers:
                    layers_sparsity.append(layer.sparsity)
                    names.append(name)
                viz.line_update(y=layers_sparsity, opts=dict(
                    xlabel='Epoch',
                    ylabel='sparsity',
                    title='KWinnersTakeAll.sparsity',
                    legend=names,
                    ytype='log',
                ))

            def hardness(viz):
                layers_hardness = []
                names = []
                for name, layer in kwta_named_layers_soft:
                    layers_hardness.append(layer.hardness)
                    names.append(name)
                viz.line_update(y=layers_hardness, opts=dict(
                    xlabel='Epoch',
                    ylabel='hardness',
                    title='KWinnersTakeAllSoft.hardness',
                    legend=names,
                    ytype='log',
                ))

            self.monitor.register_func(sparsity, hardness)

        if isinstance(self.criterion, LossFixedPattern):
            def show_fixed_patterns(viz):
                labels = sorted(self.criterion.patterns.keys())
                patterns = [self.criterion.patterns[label] for label in labels]
                patterns = torch.stack(patterns, dim=0).cpu()
                title = 'Fixed target patterns'
                viz.heatmap(patterns, win=title, opts=dict(
                    xlabel='Embedding dimension',
                    ylabel='Label',
                    title=title,
                ))

            self.monitor.register_func(show_fixed_patterns)

    def log_trainer(self):
        super().log_trainer()
        self.monitor.log(f"KWTA scheduler: {self.kwta_scheduler}")

    def _epoch_finished(self, epoch, outputs, labels):
        loss = super()._epoch_finished(epoch, outputs, labels)
        if self.kwta_scheduler is not None:
            self.kwta_scheduler.step(epoch=epoch)
        return loss

    def train_mask(self):
        image, label = super().train_mask()
        mode_saved = prepare_eval(self.model)
        with torch.no_grad():
            output = self.model(image.unsqueeze(dim=0))
        output_sorted, argsort = output[0].sort(dim=0, descending=True)
        neurons_check = min(5, len(argsort))
        for i in range(neurons_check):
            neuron_max = argsort[i]
            self.monitor.plot_mask(self.model, mask_trainer=self.mask_trainer_kwta, image=image, label=neuron_max,
                                   win_suffix=i)
        mode_saved.restore(self.model)
        return image, label

    def restore(self, checkpoint_path=None, strict=True):
        checkpoint_state = super().restore(checkpoint_path=checkpoint_path, strict=strict)
        self.kwta_scheduler.last_epoch_update = self.timer.epoch - 1
        return checkpoint_state
