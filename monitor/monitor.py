import torch
import torch.utils.data

from mighty.monitor import MonitorAutoenc


class MonitorAutoencBinary(MonitorAutoenc):

    def plot_autoencoder_binary(self, images, reconstructed,
                                reconstructed_binary, n_show=30):
        if images.shape != reconstructed.shape:
            raise ValueError("Input & reconstructed image shapes differ")
        n_show = min(images.shape[0], n_show)
        images = images[: n_show]
        reconstructed = reconstructed[: n_show]
        reconstructed_binary = reconstructed_binary[: n_show]
        if self.normalize_inverse is not None:
            images = self.normalize_inverse(images)
            reconstructed = self.normalize_inverse(reconstructed)
            # reconstructed_binary is already in [0, 1] range
        images_stacked = torch.cat(
            [images, reconstructed, reconstructed_binary], dim=0)
        images_stacked.clamp_(0, 1)
        self.viz.images(images_stacked, nrow=n_show, win='autoencoder',
            opts=dict(title="Original (Top) | Reconstructed "
                            "| Reconstructed binary"))

    def plot_reconstruction_exact(self, n_exact, n_total):
        title = "Reconstruction exact"
        self.viz.line_update([n_exact, n_total], opts=dict(
            title=title,
            xlabel="Epoch",
            ylabel="num. of exactly reconstructed",
            legend=["exact", "#samples"],
        ))
