import numpy as np
import torch
import torch.utils.data

from mighty.monitor import MonitorAutoenc, MonitorEmbedding


class MonitorEmbeddingKWTA(MonitorEmbedding):

    def clusters_heatmap(self, mean, std, save=False):
        if mean is None:
            return
        super().clusters_heatmap(mean, std, save)
        clusters_sparsity = mean.abs().mean(dim=1)  # (C,)
        clusters_sparsity.unsqueeze_(dim=1)

        n_classes = mean.shape[0]
        win = "Clusters L1 sparsity"
        opts = dict(
            title=win,
            ylabel='Label',
            rownames=list(map(str, range(n_classes))),
            columnnames=[''],
            width=200,
            height=None,
        )
        if n_classes <= self.n_classes_format_ytickstep_1:
            opts.update(ytickstep=1)

        self.viz.heatmap(clusters_sparsity.cpu(), win=win, opts=opts)


class MonitorAutoencBinary(MonitorEmbeddingKWTA, MonitorAutoenc):

    def plot_autoencoder_binary(self, images, reconstructed,
                                reconstructed_binary, n_show=10):
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
                                        "| Reconstructed binary",
                                  width=1000,
                                  height=None,
                                  ))

    def plot_reconstruction_exact(self, n_exact, n_total=None, mode='train'):
        named_metric = [(mode, n_exact)]
        if n_total is not None:
            named_metric.append((f"#total-{mode}", n_total))
        for name, val in named_metric:
            dash = 'solid' if 'total' not in name else 'dash'
            self.viz.line_update(val, opts=dict(
                title="Reconstruction exact",
                xlabel="Epoch",
                ylabel="num. of exactly reconstructed",
                dash=np.array([dash]),
            ), name=name)
