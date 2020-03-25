import torch
import torch.utils.data
from mighty.monitor import Monitor
from mighty.utils.domain import MonitorLevel


class MonitorAutoenc(Monitor):

    def epoch_finished(self, outputs=None, labels_true=None):
        self.update_mutual_info()
        for monitored_function in self.functions:
            monitored_function(self.viz)
        self.update_grad_norm()
        if self._advanced_monitoring_level.value >= MonitorLevel.SIGNAL_TO_NOISE.value:
            self.update_gradient_signal_to_noise_ratio()
        if self._advanced_monitoring_level is MonitorLevel.FULL:
            self.param_records.plot_sign_flips(self.viz)
            self.update_initial_difference()

    def plot_autoencoder(self, images, reconstructed, n_show=10):
        assert images.shape == reconstructed.shape, "Input & decoded image shape differs"
        n_show = min(images.shape[0], n_show)
        images = images[: n_show]
        if self.normalize_inverse is not None:
            images = self.normalize_inverse(images)
        reconstructed = reconstructed[: n_show]
        reconstructed = reconstructed.sigmoid()
        images_stacked = torch.cat([images, reconstructed], dim=0)
        images_stacked.clamp_(0, 1)
        self.viz.images(images_stacked, nrow=n_show, win='autoencoder', opts=dict(
            title=f"Original | Decoded",
        ))
