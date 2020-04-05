from typing import Union

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from mighty.monitor.accuracy import Accuracy
from mighty.monitor.var_online import MeanOnlineBatch
from mighty.trainer import TrainerAutoencoder
from mighty.utils.data import DataLoader, get_normalize_inverse
from monitor.accuracy import AccuracyAutoencoderBinary
from monitor.monitor import MonitorAutoencBinary
from utils import dataset_sparsity
from .kwta import TrainerEmbeddingKWTA


class TrainerAutoencoderBinary(TrainerAutoencoder, TrainerEmbeddingKWTA):

    def __init__(self, model: nn.Module, criterion: nn.Module,
                 data_loader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau] = None,
                 reconstruct_threshold: torch.Tensor = None,
                 accuracy_measure: Accuracy = AccuracyAutoencoderBinary(),
                 **kwargs):
        TrainerEmbeddingKWTA.__init__(self, model,
                                      criterion=criterion,
                                      data_loader=data_loader,
                                      optimizer=optimizer,
                                      scheduler=scheduler,
                                      accuracy_measure=accuracy_measure,
                                      **kwargs)
        if reconstruct_threshold is None:
            reconstruct_threshold = torch.linspace(0., 0.95, steps=10,
                                                   dtype=torch.float32)
        self.reconstruct_thr = reconstruct_threshold.view(1, 1, -1)
        self.dataset_sparsity = dataset_sparsity(data_loader.dataset_cls)

    def _init_monitor(self, mutual_info) -> MonitorAutoencBinary:
        normalize_inverse = get_normalize_inverse(self.data_loader.normalize)
        monitor = MonitorAutoencBinary(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=normalize_inverse
        )
        return monitor

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['pixel-error'] = MeanOnlineBatch()
        return online

    def _on_forward_pass_batch(self, input, output, labels):
        latent, reconstructed = output
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            reconstructed = reconstructed.sigmoid()

        # update pixel error
        rec_flatten = reconstructed.cpu().view(reconstructed.shape[0], -1, 1)
        rec_binary = rec_flatten >= self.reconstruct_thr  # (B, V, THR)
        input_binary = (input.cpu() > self.dataset_sparsity).view(
            input.shape[0], -1, 1)  # (B, V, 1)
        pix_miss = (rec_binary ^ input_binary).sum(dim=1, dtype=torch.float32)
        # pix_miss is of shape (B, THR)
        self.online['pixel-error'].update(pix_miss)

        super()._on_forward_pass_batch(input, output, labels)

    def _epoch_finished(self, epoch, loss):
        self.plot_autoencoder()
        self.monitor.plot_reconstruction_error(
            self.online['pixel-error'].get_mean(),
            self.reconstruct_thr.squeeze()
        )
        super()._epoch_finished(epoch, loss)

    def plot_autoencoder(self):
        input, labels = next(iter(self.data_loader.eval))
        if torch.cuda.is_available():
            input = input.cuda()
        mode_saved = self.model.training
        self.model.train(False)
        with torch.no_grad():
            latent, reconstructed = self.model(input)
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            reconstructed = reconstructed.sigmoid()

        lowest_id = self.online['pixel-error'].get_mean().argmin()
        thr_lowest = self.reconstruct_thr[0, 0, lowest_id]
        rec_binary = (reconstructed >= thr_lowest).type(torch.float32)
        self.monitor.plot_autoencoder_binary(input, reconstructed, rec_binary)
        self.model.train(mode_saved)
