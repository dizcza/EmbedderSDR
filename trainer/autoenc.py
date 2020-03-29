from typing import Union, Optional

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

from mighty.monitor.var_online import MeanOnline, MeanOnlineVector
from mighty.utils.algebra import compute_psnr
from mighty.utils.data import DataLoader
from mighty.utils.data import get_normalize_inverse
from monitor.monitor import MonitorAutoenc
from trainer.kwta import TrainerGradKWTA, KWTAScheduler
from utils import dataset_sparsity


class TrainerAutoenc(TrainerGradKWTA):


    def __init__(self, model: nn.Module, criterion: nn.Module, data_loader: DataLoader,
                 optimizer: torch.optim.Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau, None] = None,
                 kwta_scheduler: Optional[KWTAScheduler] = None,
                 accuracy_measure=None,
                 **kwargs):
        super().__init__(model, criterion=criterion, data_loader=data_loader,
                         optimizer=optimizer, scheduler=scheduler,
                         kwta_scheduler=kwta_scheduler,
                         accuracy_measure=accuracy_measure, **kwargs)
        self.reconstruct_thr = torch.linspace(0.4, 0.95, steps=10,
                                              dtype=torch.float32)
        self.reconstruct_thr = self.reconstruct_thr.view(1, 1, -1)
        self.dataset_sparsity = dataset_sparsity(data_loader.dataset_cls)


    def _init_monitor(self, mutual_info) -> MonitorAutoenc:
        normalize_inverse = get_normalize_inverse(self.data_loader.normalize)
        monitor = MonitorAutoenc(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=normalize_inverse
        )
        return monitor

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['psnr'] = MeanOnline()
        online['pixel-error'] = MeanOnlineVector()
        return online

    def _get_loss(self, input, output, labels):
        latent, reconstructed = output
        return self.criterion(reconstructed, input)

    def _on_forward_pass_batch(self, input, output, labels):
        latent, reconstructed = output
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            reconstructed = reconstructed.sigmoid()
        psnr = compute_psnr(input, reconstructed)
        self.online['psnr'].update(psnr)

        # update pixel error
        rec_flatten = reconstructed.view(reconstructed.shape[0], -1, 1)
        rec_binary = rec_flatten >= self.reconstruct_thr
        input_binary = (input > self.dataset_sparsity).view(
            input.shape[0], -1, 1)
        pix_miss = (rec_binary ^ input_binary).sum(dim=1, dtype=torch.float32)
        self.online['pixel-error'].update(pix_miss)

        super()._on_forward_pass_batch(input, latent, labels)

    def _epoch_finished(self, epoch, loss):
        self.plot_autoencoder()
        self.monitor.plot_psnr(self.online['psnr'].get_mean())
        self.monitor.plot_reconstruction_error(
            self.online['pixel-error'].get_mean(),
            self.reconstruct_thr.squeeze()
        )
        super()._epoch_finished(epoch, loss)

    def plot_autoencoder(self):
        input, labels = next(iter(self.data_loader.eval))
        self.model.eval()
        with torch.no_grad():
            latent, reconstructed = self.model(input)
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            reconstructed = reconstructed.sigmoid()
        self.monitor.plot_autoencoder(input, reconstructed)

        rec_flatten = reconstructed.view(reconstructed.shape[0], -1, 1)
        rec_binary = rec_flatten >= self.reconstruct_thr
        input_binary = (input > self.dataset_sparsity).view(
            input.shape[0], -1, 1)

        self.model.train()
