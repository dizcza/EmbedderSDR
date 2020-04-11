import warnings
from typing import Union, Optional

import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from mighty.trainer import TrainerEmbedding, TrainerGrad
from mighty.trainer.mask import MaskTrainerIndex
from mighty.utils.common import find_layers, find_named_layers
from mighty.utils.data import DataLoader
from mighty.utils.prepare import prepare_eval
from models.kwta import KWinnersTakeAllSoft, KWinnersTakeAll, SynapticScaling
from monitor.accuracy import AccuracyEmbeddingKWTA


class KWTAScheduler:
    """
    KWinnersTakeAll sparsity scheduler.
    """

    def __init__(self, model: nn.Module, step_size: int, gamma_sparsity=0.5,
                 min_sparsity=0.05,
                 gamma_hardness=2.0, max_hardness=10):
        self.kwta_layers = tuple(
            find_layers(model, layer_class=KWinnersTakeAll))
        self.step_size = step_size
        self.gamma_sparsity = gamma_sparsity
        self.min_sparsity = min_sparsity
        self.gamma_hardness = gamma_hardness
        self.max_hardness = max_hardness
        self.last_epoch_update = -1

    def need_update(self, epoch: int):
        return epoch >= self.last_epoch_update + self.step_size

    def step(self, epoch: int):
        updated = False
        if self.need_update(epoch):
            for layer in self.kwta_layers:
                if layer.sparsity is not None:
                    layer.sparsity = max(layer.sparsity * self.gamma_sparsity,
                                         self.min_sparsity)
                    updated |= layer.sparsity != self.min_sparsity
                if isinstance(layer, KWinnersTakeAllSoft):
                    layer.hardness = min(layer.hardness * self.gamma_hardness,
                                         self.max_hardness)
                    updated |= layer.hardness != self.max_hardness
            self.last_epoch_update = epoch
        return updated

    def state_dict(self):
        return {
            'last_epoch_update': self.last_epoch_update
        }

    def load_state_dict(self, state_dict: dict):
        if state_dict is not None:
            self.last_epoch_update = state_dict['last_epoch_update']

    def extra_repr(self):
        return f"step_size={self.step_size}," \
               f"Sparsity(gamma={self.gamma_sparsity}," \
               f"min={self.min_sparsity}), " \
               f"Hardness(gamma={self.gamma_hardness}," \
               f"max={self.max_hardness})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"


class TrainerEmbeddingKWTA(TrainerEmbedding):
    """
    Trainer for neural networks with k-winners-take-all (kWTA) activation
    function. If the model does not have any KWinnerTakeAll layers, it acts
    as TrainerGradEmbedding.
    """

    watch_modules = TrainerGrad.watch_modules + (KWinnersTakeAll,
                                                 SynapticScaling)

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 data_loader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau] = None,
                 kwta_scheduler: Optional[KWTAScheduler] = None,
                 accuracy_measure=AccuracyEmbeddingKWTA(),
                 env_suffix='',
                 **kwargs):
        """
        :param model: NN model
        :param criterion: loss function
        :param dataset_name: one of "MNIST", "CIFAR10", "Caltech256"
        :param optimizer: gradient-based optimizer (SGD, Adam)
        :param scheduler: learning rate scheduler
        :param kwta_scheduler: kWTA sparsity and hardness scheduler
        """
        kwta_layers = tuple(find_layers(model,
                                        layer_class=KWinnersTakeAll))
        if not kwta_layers:
            warnings.warn(
                "For models with no kWTA layer, use TrainerEmbedding")
            env_suffix = f"{env_suffix} no-kwta"
            if kwta_scheduler is not None:
                warnings.warn("Turning off KWTAScheduler, because the model "
                              "does not have kWTA layers.")
                kwta_scheduler = None
            if isinstance(accuracy_measure, AccuracyEmbeddingKWTA):
                warnings.warn(
                    "Setting AccuracyEmbeddingKWTA.sparsity to None, "
                    "because the model does not have kWTA layers.")
                accuracy_measure.sparsity = None
        elif len(kwta_layers) > 1:
            raise ValueError("Only 1 kWTA layer per model is accepted.")
        kwta = kwta_layers[0]
        if getattr(kwta, "threshold", None) is not None:
            # kwta-soft with a threshold
            env_suffix = f"{env_suffix} threshold"

        super().__init__(model=model,
                         criterion=criterion,
                         data_loader=data_loader,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         accuracy_measure=accuracy_measure,
                         env_suffix=env_suffix,
                         **kwargs)
        self.kwta_scheduler = kwta_scheduler
        self._update_accuracy_state()

    def has_kwta(self) -> bool:
        return any(find_layers(self.model, KWinnersTakeAll))

    def monitor_functions(self):
        super().monitor_functions()
        if not self.has_kwta():
            return

        def kwta_centroids(viz):
            if not self.accuracy_measure.is_fit:
                return
            class_centroids = self.accuracy_measure.centroids
            win = "kWTA class centroids heatmap"
            opts = dict(
                title=f"{win}. Epoch {self.timer.epoch}",
                xlabel='Embedding dimension',
                ylabel='Label',
                rownames=[str(i) for i in range(class_centroids.shape[0])],
            )
            if class_centroids.shape[0] <= \
                    self.monitor.n_classes_format_ytickstep_1:
                opts.update(ytickstep=1)
            viz.heatmap(class_centroids, win=win, opts=opts)

        self.monitor.register_func(kwta_centroids)

        if self.kwta_scheduler is None:
            return

        kwta_named_layers = tuple(find_named_layers(
            self.model, layer_class=KWinnersTakeAll))
        kwta_named_layers_soft = tuple(find_named_layers(
            self.model, layer_class=KWinnersTakeAllSoft))

        def sparsity(viz):
            layers_sparsity = []
            names = []
            for name, layer in kwta_named_layers:
                if layer.sparsity is not None:
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

        _, kwta = kwta_named_layers[0]
        if self.kwta_scheduler.min_sparsity != kwta.sparsity:
            self.monitor.register_func(sparsity)

        self.monitor.register_func(hardness)

    def log_trainer(self):
        super().log_trainer()
        if self.kwta_scheduler:
            self.monitor.log(repr(self.kwta_scheduler))

    def _epoch_finished(self, epoch, loss):
        if self.kwta_scheduler is not None:
            updated = self.kwta_scheduler.step(epoch=epoch)
            if updated:
                # save the activations heatmap before the update
                self.monitor.clusters_heatmap(
                    *self.online['clusters'].get_mean_std(),
                    save=True)
        self._update_accuracy_state()
        super()._epoch_finished(epoch, loss)

    def _update_accuracy_state(self):
        if not self.has_kwta() or not isinstance(self.accuracy_measure,
                                                 AccuracyEmbeddingKWTA):
            return
        # only 1 kWTA per model is accepted
        kwta_layer = next(find_layers(self.model, KWinnersTakeAll))
        sparsity = kwta_layer.sparsity
        if sparsity is None:
            # KWinnersTakeAllSoft with threshold
            # online['sparsity'].get_mean() returns None before the training
            # is started, but it's fine
            sparsity = self.online['sparsity'].get_mean()
        self.accuracy_measure.sparsity = sparsity

    def train_mask(self):
        image, label = super().train_mask()
        if not self.has_kwta():
            return image, label
        mask_trainer_kwta = MaskTrainerIndex(image_shape=image.shape)
        mode_saved = prepare_eval(self.model)
        with torch.no_grad():
            output = self.model(image.unsqueeze(dim=0))
        output_sorted, argsort = output[0].sort(dim=0, descending=True)
        neurons_check = min(5, len(argsort))
        for i in range(neurons_check):
            neuron_max = argsort[i]
            self.monitor.plot_mask(self.model,
                                   mask_trainer=mask_trainer_kwta,
                                   image=image,
                                   label=neuron_max,
                                   win_suffix=i)
        mode_saved.restore(self.model)
        return image, label

    def state_dict(self):
        state = super().state_dict()
        if self.has_kwta() and self.kwta_scheduler is not None:
            state['kwta_scheduler'] = self.kwta_scheduler.state_dict()
        return state

    def restore(self, checkpoint_path=None, strict=True):
        checkpoint_state = super().restore(checkpoint_path, strict=strict)
        if self.has_kwta() and checkpoint_state is not None:
            kwta_scheduler = checkpoint_state.get('kwta_scheduler', None)
            self.kwta_scheduler.load_state_dict(kwta_scheduler)
        return checkpoint_state
