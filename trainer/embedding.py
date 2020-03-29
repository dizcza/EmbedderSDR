from typing import Union

import torch.nn as nn
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from mighty.monitor.var_online import MeanOnline, MeanOnlineVector, \
    VarianceOnline
from mighty.trainer.gradient import TrainerGrad
from mighty.utils.data import DataLoader
from mighty.utils.data import get_normalize_inverse
from monitor.accuracy import AccuracyEmbedding
from monitor.monitor import MonitorEmbedding


class TrainerEmbedding(TrainerGrad):
    """
    Operates on embedding vectors.
    """

    def __init__(self, model: nn.Module, criterion: nn.Module,
                 data_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[
                     _LRScheduler, ReduceLROnPlateau, None] = None,
                 accuracy_measure=AccuracyEmbedding(),
                 **kwargs):
        """
        :param model: NN model
        :param criterion: loss function
        :param dataset_name: one of "MNIST", "CIFAR10", "Caltech256"
        :param optimizer: gradient-based optimizer (SGD, Adam)
        :param scheduler: learning rate scheduler
        :param kwta_scheduler: kWTA sparsity and hardness scheduler
        """
        if not isinstance(accuracy_measure, AccuracyEmbedding):
            raise ValueError("'accuracy_measure' must be of instance "
                             "AccuracyEmbedding")
        super().__init__(model=model,
                         criterion=criterion,
                         data_loader=data_loader,
                         optimizer=optimizer,
                         scheduler=scheduler,
                         accuracy_measure=accuracy_measure,
                         **kwargs)

    def _init_monitor(self, mutual_info):
        normalize_inverse = get_normalize_inverse(self.data_loader.normalize)
        monitor = MonitorEmbedding(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=normalize_inverse
        )
        return monitor

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['sparsity'] = MeanOnline()  # scalar
        online['firing_rate'] = MeanOnlineVector()  # (V,) vector
        online['clusters'] = VarianceOnline()  # (C, V) tensor
        return online

    def _on_forward_pass_batch(self, input, output, labels):
        super()._on_forward_pass_batch(input, output, labels)
        sparsity = output.norm(p=1, dim=1).mean() / output.shape[1]
        self.online['sparsity'].update(sparsity)
        self.online['firing_rate'].update(output)

        # update clusters
        class_centroids = []
        for label in sorted(labels.unique(sorted=True)):
            outputs_label = output[labels == label]
            class_centroids.append(outputs_label.mean(dim=0))
        class_centroids = torch.stack(class_centroids, dim=0)  # (C, V)
        self.online['clusters'].update(class_centroids)

    def _epoch_finished(self, epoch, loss):
        self.monitor.update_sparsity(self.online['sparsity'].get_mean(),
                                     mode='train')
        self.monitor.update_firing_rate(self.online['firing_rate'].get_mean())
        self.monitor.clusters_heatmap(*self.online['clusters'].get_mean_std())
        super()._epoch_finished(epoch, loss)
