from typing import Union, Optional
import warnings

import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
from torch.optim.optimizer import Optimizer

from mighty.monitor.accuracy import Accuracy
from mighty.utils.var_online import MeanOnlineBatch, SumOnlineBatch
from mighty.trainer import TrainerAutoencoder
from mighty.utils.common import input_from_batch
from mighty.utils.data import DataLoader
from monitor.accuracy import AccuracyEmbeddingKWTA
from monitor.monitor import MonitorAutoencoderBinary
from utils import dataset_mean
from .kwta import InterfaceKWTA, KWTAScheduler


class Reconstruct:
    def __init__(self, threshold=None, dataset_mean=0.):
        if threshold is None:
            threshold = torch.linspace(0., 0.95, steps=10, dtype=torch.float32)
        if torch.cuda.is_available():
            threshold = threshold.cuda()
        self.threshold = threshold
        self.dataset_mean = dataset_mean
        self.is_fixed = False
        self.optimal_id = len(self.threshold) // 2

    def __repr__(self):
        fixed_thr = self.threshold[self.optimal_id] if self.is_fixed else None
        return f"{self.__class__.__name__}(dataset_mean={self.dataset_mean}" \
               f", fixed_threshold={fixed_thr})"

    @property
    def threshold_optimal(self):
        return self.threshold[self.optimal_id]

    def fix_threshold(self, new_threshold: float):
        self.is_fixed = True
        self.threshold = torch.tensor([new_threshold],
                                      device=self.threshold.device)
        self.optimal_id = 0

    def compute(self, input: torch.Tensor, reconstructed: torch.Tensor):
        # input and reconstructed are of shape (B, C, H, W)
        if input.ndim == 3:
            input = input.unsqueeze(dim=0)
            reconstructed = reconstructed.unsqueeze(dim=0)
        # update pixel error
        rec_flatten = reconstructed.view(reconstructed.shape[0], -1, 1)
        threshold = self.threshold.view(1, 1, -1)  # (B, In, THR)
        rec_binary = rec_flatten >= threshold
        input_binary = input > self.dataset_mean
        input_binary = input_binary.view(input.shape[0], -1, 1)  # (B, In, 1)
        # (B, THR)
        pix_miss = (rec_binary ^ input_binary).sum(dim=1, dtype=torch.float32)
        correct = pix_miss[:, self.optimal_id] == 0  # (B,)
        return pix_miss, correct

    def update(self, pixel_error: torch.Tensor):
        if not self.is_fixed:
            self.optimal_id = pixel_error.argmin()


class TrainerAutoencoderBinary(InterfaceKWTA, TrainerAutoencoder):
    best_score_type = "accuracy autoencoder"

    def __init__(self, model: nn.Module, criterion: nn.Module,
                 data_loader: DataLoader,
                 optimizer: Optimizer,
                 scheduler: Union[_LRScheduler, ReduceLROnPlateau] = None,
                 reconstruct_threshold: torch.Tensor = None,
                 kwta_scheduler: Optional[KWTAScheduler] = None,
                 accuracy_measure: Accuracy = AccuracyEmbeddingKWTA(),
                 **kwargs):
        TrainerAutoencoder.__init__(self, model,
                                    criterion=criterion,
                                    data_loader=data_loader,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    accuracy_measure=accuracy_measure,
                                    **kwargs)
        InterfaceKWTA.__init__(self, kwta_scheduler)
        self.reconstruct = Reconstruct(reconstruct_threshold,
                                       dataset_mean(data_loader))

    def _init_monitor(self, mutual_info) -> MonitorAutoencoderBinary:
        monitor = MonitorAutoencoderBinary(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=self.data_loader.normalize_inverse
        )
        return monitor

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['pixel-error'] = MeanOnlineBatch()
        online['reconstruct-exact-train'] = SumOnlineBatch()
        online['reconstruct-exact-test'] = SumOnlineBatch()
        return online

    def fix_reconstruct_threshold(self, new_threshold):
        self.reconstruct.fix_threshold(new_threshold)

    def log_trainer(self):
        super().log_trainer()
        self.monitor.log(self.reconstruct)

    def _on_forward_pass_batch(self, batch, output, train):
        input = input_from_batch(batch)
        latent, reconstructed = output
        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            reconstructed = reconstructed.sigmoid()

        if self.data_loader.normalize_inverse is not None:
            warnings.warn("'normalize_inverse' is not None. Applying it "
                          "to count reconstructed pixels")
            input = self.data_loader.normalize_inverse(input)
            reconstructed = self.data_loader.normalize_inverse(reconstructed)

        pix_miss, correct = self.reconstruct.compute(input, reconstructed)
        if train:
            # update only for train
            # pix_miss is of shape (B, THR)
            self.online['pixel-error'].update(pix_miss.cpu())

        fold = 'train' if train else 'test'
        self.online[f'reconstruct-exact-{fold}'].update(correct.cpu())

        super()._on_forward_pass_batch(batch, output, train)

    def _epoch_finished(self, loss):
        self.reconstruct.update(self.online['pixel-error'].get_mean())
        self.monitor.plot_reconstruction_error(
            self.online['pixel-error'].get_mean(),
            self.reconstruct.threshold,
            self.reconstruct.optimal_id
        )
        for fold in ('train', 'test'):
            n_exact = self.online[f'reconstruct-exact-{fold}'].get_sum()
            n_total = self.online[f'reconstruct-exact-{fold}'].count
            if fold == 'train':
                accuracy = n_exact / float(n_total)
                self.update_best_score(accuracy)
            self.monitor.plot_reconstruction_exact(n_exact=n_exact,
                                                   n_total=n_total,
                                                   mode=fold)
        super()._epoch_finished(loss)

    def _plot_autoencoder(self, batch, reconstructed, mode='train'):
        input = input_from_batch(batch)
        thr_lowest = self.reconstruct.threshold_optimal
        rec_binary = (reconstructed >= thr_lowest).float()
        self.monitor.plot_autoencoder_binary(input, reconstructed, rec_binary,
                                             mode=mode)
