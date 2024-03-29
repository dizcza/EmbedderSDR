import warnings
from typing import Union, Optional

import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from mighty.trainer import TrainerEmbedding
from mighty.trainer.mask import MaskTrainer
from mighty.trainer.trainer import Trainer
from mighty.utils.common import find_layers, find_named_layers, batch_to_cuda
from mighty.utils.data import DataLoader
from mighty.utils.prepare import prepare_eval
from models.kwta import KWinnersTakeAllSoft, KWinnersTakeAll, \
    SynapticScaling, SparsityPredictor
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
        self.epoch = 0

    def need_update(self):
        return self.epoch > 0 and self.epoch % self.step_size == 0

    def step(self, epoch=None):
        updated = False
        if epoch:
            self.epoch = epoch
        else:
            # this function is called just _before_ the completion of an epoch
            # in the _epoch_finished() function
            self.epoch += 1
        if self.need_update():
            exponent = self.epoch / self.step_size
            for layer in self.kwta_layers:
                if isinstance(layer.sparsity, float):
                    layer.sparsity = max(
                        layer.sparsity * self.gamma_sparsity ** exponent,
                        self.min_sparsity
                    )
                    updated |= layer.sparsity != self.min_sparsity
                if isinstance(layer, KWinnersTakeAllSoft):
                    layer.hardness = min(
                        layer.hardness * self.gamma_hardness ** exponent,
                        self.max_hardness
                    )
                    updated |= layer.hardness != self.max_hardness
        return updated

    def state_dict(self):
        # TODO: kwta scheduler epoch differs from the timer
        return {
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict: dict):
        for layer in self.kwta_layers:
            if isinstance(layer.sparsity, float) \
                    and layer.sparsity < self.min_sparsity:
                warnings.warn(f"Reset kWTA.sparsity ({layer.sparsity}) "
                              f"to the minimum ({self.min_sparsity})")
                layer.sparsity = self.min_sparsity
            if isinstance(layer, KWinnersTakeAllSoft) \
                    and layer.hardness > self.max_hardness:
                warnings.warn(f"Reset kWTA.hardness ({layer.hardness}) "
                              f"to the maximum ({self.max_hardness})")
                layer.hardness = self.max_hardness
        if state_dict is not None:
            self.step(state_dict['epoch'])

    def extra_repr(self):
        return f"step_size={self.step_size}," \
               f"Sparsity(gamma={self.gamma_sparsity}," \
               f"min={self.min_sparsity}), " \
               f"Hardness(gamma={self.gamma_hardness}," \
               f"max={self.max_hardness})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"


class InterfaceKWTA(Trainer):

    def __init__(self, kwta_scheduler: Optional[KWTAScheduler] = None):
        self.watch_modules = self.watch_modules + (KWinnersTakeAll,
                                                   SynapticScaling)
        layers_ordered = self.get_layers_ordered(
            ignore_children=KWinnersTakeAll)
        self.kwta_layers = tuple(layer for layer in layers_ordered
                                 if isinstance(layer, KWinnersTakeAll))
        # self.kwta_layers = tuple(find_layers(self.model, KWinnersTakeAll))
        if not self.has_kwta():
            warnings.warn(
                "For models with no kWTA layer, use TrainerEmbedding")
            self.env_name = f"{self.env_name} no-kwta"
            if kwta_scheduler is not None:
                warnings.warn("Turning off KWTAScheduler, because the model "
                              "does not have kWTA layers.")
                kwta_scheduler = None
            if isinstance(self.accuracy_measure, AccuracyEmbeddingKWTA):
                warnings.warn(
                    "Setting AccuracyEmbeddingKWTA.sparsity to None, "
                    "because the model does not have kWTA layers.")
                self.accuracy_measure.sparsity = None

        kwta_soft = tuple(find_layers(self.model, KWinnersTakeAllSoft))
        if any(kwta.threshold is not None for kwta in kwta_soft):
            # kwta-soft with a threshold
            self.env_name = f"{self.env_name} threshold"
        if any(isinstance(kwta.sparsity, SparsityPredictor)
               for kwta in kwta_soft):
            self.env_name = f"{self.env_name} sparsity-predictor"

        self.kwta_scheduler = kwta_scheduler
        self._update_accuracy_state()
        self._post_init_monitor()

    def has_kwta(self) -> bool:
        return len(self.kwta_layers) > 0

    def get_layers_ordered(self, ignore_layers=(nn.Sequential,),
                           ignore_children=()):
        ignore_layers = ignore_layers + (type(self.model),)
        hooks = []
        layers_ordered = []

        def register_hooks(a_model: nn.Module):
            children = tuple(a_model.children())
            if any(children) and not isinstance(a_model, ignore_children):
                for layer in children:
                    register_hooks(layer)
            if not isinstance(a_model, ignore_layers):
                handle = a_model.register_forward_pre_hook(append_layer)
                hooks.append(handle)

        def append_layer(layer, tensor_input):
            layers_ordered.append(layer)

        register_hooks(self.model)

        batch = self.data_loader.sample()
        batch = batch_to_cuda(batch)
        with torch.no_grad():
            # start hooking
            self._forward(batch)
            # end hooking

        for handle in hooks:
            handle.remove()

        return layers_ordered

    def _update_accuracy_state(self):
        if not self.has_kwta() or not isinstance(self.accuracy_measure,
                                                 AccuracyEmbeddingKWTA):
            return
        kwta_last = self.kwta_layers[-1]
        sparsity = kwta_last.sparsity
        if sparsity is None:
            # KWinnersTakeAllSoft with threshold
            # online['sparsity'].get_mean() returns None before the training
            # is started, but it's fine
            sparsity = self.online['sparsity'].get_mean()
        self.accuracy_measure.sparsity = sparsity

    def _post_init_monitor(self):
        # hack Monitor update_sparsity() function
        if self.kwta_scheduler is None:
            self.monitor.update_sparsity = lambda *args, **kwargs: None
            return
        sparsities = tuple(kwta.sparsity for kwta in self.kwta_layers)
        if all(isinstance(sparsity, float) for sparsity in sparsities):
            sparsities = set(sparsities)
            if len(sparsities) == 1:
                sparsity = next(iter(sparsities))
                if np.isclose(sparsity, self.kwta_scheduler.min_sparsity):
                    # sparsity is not going to change, do nothing
                    self.monitor.update_sparsity = lambda *args, **kwargs: None

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
                if isinstance(layer.sparsity, float):
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

        if any(self.kwta_scheduler.min_sparsity != kwta.sparsity
               for _, kwta in kwta_named_layers):
            self.monitor.register_func(sparsity)

        if any(self.kwta_scheduler.max_hardness != kwta.hardness
               for _, kwta in kwta_named_layers_soft):
            self.monitor.register_func(hardness)

    def log_trainer(self):
        super().log_trainer()
        if self.kwta_scheduler:
            self.monitor.log(repr(self.kwta_scheduler))

    def _epoch_finished(self, loss):
        if self.kwta_scheduler is not None:
            updated = self.kwta_scheduler.step()
            if updated:
                # save the activations heatmap before the update
                self.monitor.clusters_heatmap(
                    self.online['clusters'].get_mean(),
                    save=True)
        self._update_accuracy_state()
        super()._epoch_finished(loss)

    def train_mask(self):
        image, label = super().train_mask()
        if not self.has_kwta():
            return image, label
        mask_trainer_kwta = MaskTrainer(image_shape=image.shape)
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

    def restore(self, checkpoint_path=None, best=False, strict=True):
        checkpoint_state = super().restore(checkpoint_path, best=best,
                                           strict=strict)
        if self.has_kwta() \
                and self.kwta_scheduler is not None \
                and checkpoint_state is not None:
            kwta_scheduler = checkpoint_state.get('kwta_scheduler', None)
            self.kwta_scheduler.load_state_dict(kwta_scheduler)
        return checkpoint_state


class TrainerEmbeddingKWTA(InterfaceKWTA, TrainerEmbedding):
    """
    Trainer for neural networks with k-winners-take-all (kWTA) activation
    function. If the model does not have any KWinnerTakeAll layers, it acts
    as TrainerGradEmbedding.
    """

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
        TrainerEmbedding.__init__(self, model=model,
                                  criterion=criterion,
                                  data_loader=data_loader,
                                  optimizer=optimizer,
                                  scheduler=scheduler,
                                  accuracy_measure=accuracy_measure,
                                  env_suffix=env_suffix,
                                  **kwargs)
        self.kwta_scheduler = kwta_scheduler
        InterfaceKWTA.__init__(self, kwta_scheduler)
