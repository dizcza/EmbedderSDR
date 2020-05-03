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
                                reconstructed_binary, *tensors, labels=(),
                                normalize_inverse=True, n_show=10):
        labels = ['Reconstructed binary', *labels]
        if normalize_inverse and self.normalize_inverse is not None:
            images = self.normalize_inverse(images)
            reconstructed = self.normalize_inverse(reconstructed)
            tensors = map(self.normalize_inverse, tensors)
            # reconstructed_binary is already in [0, 1] range
        self.plot_autoencoder(images, reconstructed, reconstructed_binary,
                              *tensors, labels=labels, normalize_inverse=False,
                              n_show=n_show)

    def plot_reconstruction_exact(self, n_exact, n_total=None, mode='train'):
        if n_total is not None:
            accuracy = n_exact / float(n_total)
            self.update_accuracy(accuracy=accuracy, mode=mode)
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
