import torch
import torch.utils.data
from mighty.utils.data import get_normalize_inverse

from monitor.monitor import MonitorAutoenc
from trainer.kwta import TrainerGradKWTA


class TrainerAutoenc(TrainerGradKWTA):

    def _init_monitor(self, mutual_info):
        normalize_inverse = get_normalize_inverse(self.data_loader.normalize)
        monitor = MonitorAutoenc(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=normalize_inverse
        )
        return monitor

    def _get_loss(self, input, output, labels):
        latent, reconstructed = output
        return self.criterion(reconstructed, input)

    def _on_forward_pass_batch(self, input, output, labels):
        latent, reconstructed = output
        super()._on_forward_pass_batch(input, latent, labels)

    def _epoch_finished(self, epoch, loss):
        super()._epoch_finished(epoch, loss)
        self.plot_autoencoder()

    def plot_autoencoder(self):
        images, _ = next(iter(self.data_loader.eval))
        self.model.eval()
        with torch.no_grad():
            latent, reconstructed = self.model(images)
        self.monitor.plot_autoencoder(images, reconstructed)
        self.model.train()
