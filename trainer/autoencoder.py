from typing import Union, Optional

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from mighty.monitor.accuracy import Accuracy
from mighty.monitor.var_online import MeanOnlineBatch, SumOnlineBatch
from mighty.trainer import TrainerAutoencoder
from mighty.utils.common import input_from_batch, batch_to_cuda
from mighty.utils.data import DataLoader
from monitor.accuracy import AccuracyAutoencoderBinary
from monitor.monitor import MonitorAutoencBinary
from utils import dataset_mean
from .kwta import InterfaceKWTA, KWTAScheduler


class TrainerAutoencoderBinary(InterfaceKWTA, TrainerAutoencoder):

    def __init__(self, model: nn.Module, criterion: nn.Module,
                 data_loader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau] = None,
                 reconstruct_threshold: torch.Tensor = None,
                 kwta_scheduler: Optional[KWTAScheduler] = None,
                 accuracy_measure: Accuracy = AccuracyAutoencoderBinary(),
                 **kwargs):
        TrainerAutoencoder.__init__(self, model,
                                      criterion=criterion,
                                      data_loader=data_loader,
                                      optimizer=optimizer,
                                      scheduler=scheduler,
                                      accuracy_measure=accuracy_measure,
                                      **kwargs)
        InterfaceKWTA.__init__(self, kwta_scheduler)
        if reconstruct_threshold is None:
            reconstruct_threshold = torch.linspace(0., 0.95, steps=10,
                                                   dtype=torch.float32)
        if torch.cuda.is_available():
            reconstruct_threshold = reconstruct_threshold.cuda()
        self.reconstruct_thr = reconstruct_threshold.view(1, 1, -1)

        # the optimal threshold id; will be changed later
        self.thr_opt_id = len(self.reconstruct_thr) // 2
        self.dataset_sparsity = dataset_mean(data_loader)

    def _init_monitor(self, mutual_info) -> MonitorAutoencBinary:
        monitor = MonitorAutoencBinary(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=self.data_loader.normalize_inverse
        )
        return monitor

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['pixel-error'] = MeanOnlineBatch()
        online['reconstruct-exact'] = SumOnlineBatch()
        return online

    def _on_forward_pass_batch(self, batch, output):
        input = input_from_batch(batch)
        latent, reconstructed = output
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            reconstructed = reconstructed.sigmoid()

        # update pixel error
        rec_flatten = reconstructed.view(reconstructed.shape[0], -1, 1)
        rec_binary = rec_flatten >= self.reconstruct_thr  # (B, In, THR)
        if self.data_loader.normalize_inverse is not None:
            input = self.data_loader.normalize_inverse(input)
        input_binary = input > self.dataset_sparsity
        input_binary = input_binary.view(input.shape[0], -1, 1)  # (B, In, 1)
        pix_miss = (rec_binary ^ input_binary).sum(dim=1, dtype=torch.float32)
        # pix_miss is of shape (B, THR)
        self.online['pixel-error'].update(pix_miss.cpu())

        correct = pix_miss[:, self.thr_opt_id] == 0
        self.online['reconstruct-exact'].update(correct.cpu())

        super()._on_forward_pass_batch(batch, output)

    def _epoch_finished(self, loss):
        self.thr_opt_id = self.online['pixel-error'].get_mean().argmin()
        self.monitor.plot_reconstruction_error(
            self.online['pixel-error'].get_mean(),
            self.reconstruct_thr.squeeze()
        )
        n_exact = self.online['reconstruct-exact'].get_sum()
        n_total = self.online['reconstruct-exact'].count
        self.monitor.plot_reconstruction_exact(n_exact=n_exact,
                                               n_total=n_total)
        super()._epoch_finished(loss)

    def plot_autoencoder(self):
        batch = self.data_loader.sample()
        batch = batch_to_cuda(batch)
        input = input_from_batch(batch)
        mode_saved = self.model.training
        self.model.train(False)
        with torch.no_grad():
            latent, reconstructed = self._forward(batch)
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            reconstructed = reconstructed.sigmoid()

        thr_lowest = self.reconstruct_thr[0, 0, self.thr_opt_id]
        rec_binary = (reconstructed >= thr_lowest).float()
        self.monitor.plot_autoencoder_binary(input, reconstructed, rec_binary)
        self.model.train(mode_saved)
