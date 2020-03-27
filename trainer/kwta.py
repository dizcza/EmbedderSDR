import warnings
from typing import Union, Optional

import torch.nn as nn
import torch.utils.data
from mighty.monitor.var_online import MeanOnline, MeanOnlineVector
from mighty.trainer.gradient import TrainerGrad
from mighty.trainer.mask import MaskTrainerIndex
from mighty.utils.common import find_layers, find_named_layers
from mighty.utils.data import DataLoader
from mighty.utils.prepare import prepare_eval
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from mighty.utils.data import get_normalize_inverse

from models.kwta import KWinnersTakeAllSoft, KWinnersTakeAll, SynapticScaling
from monitor.accuracy import AccuracyEmbeddingKWTA
from monitor.monitor import MonitorKWTA


class KWTAScheduler:
    """
    KWinnersTakeAll sparsity scheduler.
    """

    def __init__(self, model: nn.Module, step_size: int, gamma_sparsity=0.5, min_sparsity=0.05,
                 gamma_hardness=2.0, max_hardness=10):
        self.kwta_layers = tuple(find_layers(model, layer_class=KWinnersTakeAll))
        self.step_size = step_size
        self.gamma_sparsity = torch.as_tensor(gamma_sparsity, dtype=torch.float32)
        self.min_sparsity = torch.as_tensor(min_sparsity, dtype=torch.float32)
        self.gamma_hardness = torch.as_tensor(gamma_hardness, dtype=torch.float32)
        self.max_hardness = torch.as_tensor(max_hardness, dtype=torch.float32)
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

    def state_dict(self):
        return {
            'last_epoch_update': self.last_epoch_update
        }

    def load_state_dict(self, state_dict: dict):
        self.last_epoch_update = state_dict['last_epoch_update']

    def extra_repr(self):
        return f"step_size={self.step_size}, Sparsity(gamma={self.gamma_sparsity}, min={self.min_sparsity}), " \
               f"Hardness(gamma={self.gamma_hardness}, max={self.max_hardness})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"


class TrainerGradKWTA(TrainerGrad):
    """
    Trainer for neural networks with k-winners-take-all (kWTA) activation function.
    If the model does not have any KWinnerTakeAll layers, it works just as TrainerGrad.
    """

    watch_modules = TrainerGrad.watch_modules + (KWinnersTakeAll, SynapticScaling)

    def __init__(self, model: nn.Module, criterion: nn.Module, data_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau, None] = None,
                 kwta_scheduler: Optional[KWTAScheduler] = None,
                 accuracy_measure=None,
                 **kwargs):
        """
        :param model: NN model
        :param criterion: loss function
        :param dataset_name: one of "MNIST", "CIFAR10", "Caltech256"
        :param optimizer: gradient-based optimizer (SGD, Adam)
        :param scheduler: learning rate scheduler
        :param kwta_scheduler: kWTA sparsity and hardness scheduler
        """
        if accuracy_measure is None:
            accuracy_measure = AccuracyEmbeddingKWTA()
        if not isinstance(accuracy_measure, AccuracyEmbeddingKWTA):
            raise ValueError("'accuracy_measure' must be of instance "
                             "AccuracyEmbeddingKWTA")
        super().__init__(model=model,
                         criterion=criterion,
                         data_loader=data_loader,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         accuracy_measure=accuracy_measure,
                         **kwargs)
        kwta_layers = tuple(find_layers(self.model,
                                        layer_class=KWinnersTakeAll))
        if len(kwta_layers) == 0:
            raise ValueError("When a model has no kWTA layer, use TrainerGrad")
        elif len(kwta_layers) > 1:
            raise ValueError("Only 1 kWTA layer is accepted per model.")
        self.kwta_scheduler = kwta_scheduler
        self.mask_trainer_kwta = MaskTrainerIndex(
            image_shape=self.mask_trainer.image_shape)
        self._update_accuracy_state()

    def _init_monitor(self, mutual_info):
        normalize_inverse = get_normalize_inverse(self.data_loader.normalize)
        monitor = MonitorKWTA(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=normalize_inverse
        )
        return monitor

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['sparsity'] = MeanOnline()
        online['firing_rate'] = MeanOnlineVector()
        return online

    def monitor_functions(self):
        super().monitor_functions()

        def kwta_centroids(viz):
            class_centroids = self.accuracy_measure.centroids
            win = "kWTA class centroids heatmap"
            opts = dict(
                title=f"{win}. Epoch {self.timer.epoch}",
                xlabel='Embedding dimension',
                ylabel='Label',
                rownames=[str(i) for i in range(class_centroids.shape[0])],
            )
            if class_centroids.shape[0] <= self.monitor.n_classes_format_ytickstep_1:
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

        self.monitor.register_func(sparsity)
        self.monitor.register_func(hardness)

    def log_trainer(self):
        super().log_trainer()
        self.monitor.log(repr(self.kwta_scheduler))

    def _on_forward_pass_batch(self, input, output, labels):
        super()._on_forward_pass_batch(input, output, labels)
        sparsity = output.norm(p=1, dim=1).mean() / output.shape[1]
        self.online['sparsity'].update(sparsity)
        self.online['firing_rate'].update(output)

    def _epoch_finished(self, epoch, loss):
        super()._epoch_finished(epoch, loss)
        if self.kwta_scheduler is not None:
            self.kwta_scheduler.step(epoch=epoch)
        self._update_accuracy_state()
        self.monitor.update_sparsity(self.online['sparsity'].get_mean(),
                                     mode='train')
        self.monitor.update_firing_rate(self.online['firing_rate'].get_mean())
        for online_measure in self.online.values():
            online_measure.reset()

    def _update_accuracy_state(self):
        if not isinstance(self.accuracy_measure, AccuracyEmbeddingKWTA):
            return
        sparsities = set()
        for kwta_layer in find_layers(self.model, layer_class=KWinnersTakeAll):
            sparsities.add(kwta_layer.sparsity)
        sparsities = sorted(sparsities)
        if len(sparsities) > 1:
            # irrelevant now
            warnings.warn(f"Found {len(sparsities)} layers with different sparsities: {sparsities}. "
                          f"Chose the lowest one for {self.accuracy_measure.__class__.__name__}.")
        # finally, update accuracy_measure sparsity here
        self.accuracy_measure.sparsity = sparsities[0]

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

    def state_dict(self):
        state = super().state_dict()
        state['kwta_scheduler'] = self.kwta_scheduler.state_dict()
        return state

    def restore(self, checkpoint_path=None, strict=True):
        checkpoint_state = super().restore(checkpoint_path=checkpoint_path, strict=strict)
        if checkpoint_state is not None:
            self.kwta_scheduler.load_state_dict(checkpoint_state['kwta_scheduler'])
        return checkpoint_state
