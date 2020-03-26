import numpy as np
import torch
import torch.utils.data
import torch.utils.data
import torch.utils.data
from mighty.monitor import Monitor
from mighty.monitor.batch_timer import ScheduleStep
from sklearn.metrics import pairwise


class MonitorKWTA(Monitor):

    def update_sparsity(self, sparsity: float, mode: str):
        # L1 sparsity
        self.viz.line_update(y=sparsity, opts=dict(
            xlabel='Epoch',
            ylabel='L1 norm / size',
            title='Output sparsity',
        ), name=mode)

    def activations_heatmap(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        We'd like the last layer activations heatmap to be different for each
        corresponding label.

        :param outputs: the last layer activations
        :param labels: corresponding labels
        """

        def compute_manhattan_dist(tensor: torch.FloatTensor) -> float:
            l1_dist = pairwise.manhattan_distances(tensor.cpu())
            upper_triangle_idx = np.triu_indices_from(l1_dist, k=1)
            l1_dist = l1_dist[upper_triangle_idx].mean()
            return l1_dist

        outputs = outputs.detach()
        class_centroids = []
        std_centroids = []
        label_names = []
        for label in sorted(labels.unique()):
            outputs_label = outputs[labels == label]
            std_centroids.append(outputs_label.std(dim=0))
            class_centroids.append(outputs_label.mean(dim=0))
            label_names.append(str(label.item()))
        win = "Last layer activations heatmap"
        class_centroids = torch.stack(class_centroids, dim=0)
        std_centroids = torch.stack(std_centroids, dim=0)
        opts = dict(
            title=f"{win}. Epoch {self.timer.epoch}",
            xlabel='Embedding dimension',
            ylabel='Label',
            rownames=range(class_centroids.shape[0]),
        )
        if class_centroids.shape[0] <= self.n_classes_format_ytickstep_1:
            opts.update(ytickstep=1)
        self.viz.heatmap(class_centroids, win=win, opts=opts)
        self.save_heatmap(class_centroids, win=win, opts=opts)
        normalizer = class_centroids.norm(p=1, dim=1).mean()
        outer_distance = compute_manhattan_dist(class_centroids) / normalizer
        std = std_centroids.norm(p=1, dim=1).mean() / normalizer
        self.viz.line_update(y=[outer_distance.item(), std.item()], opts=dict(
            xlabel='Epoch',
            ylabel='Mean pairwise distance (normalized)',
            legend=['inter-distance', 'intra-STD'],
            title='How much do patterns differ in L1 measure?',
        ))

    @ScheduleStep(epoch_step=20)
    def save_heatmap(self, heatmap, win, opts):
        self.viz.heatmap(heatmap, win=f"{win}. Epoch {self.timer.epoch}",
                         opts=opts)

    def update_firing_rate(self, rate: torch.Tensor):
        rate = rate.unsqueeze(dim=0)
        title = 'Neuron firing rate'
        self.viz.heatmap(rate, win=title, opts=dict(
            title=title,
            xlabel='Embedding dimension',
            rownames=['Last layer'],
            width=None,
            height=200,
        ))


class MonitorAutoenc(MonitorKWTA):

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
        self.viz.images(images_stacked, nrow=n_show, win='autoencoder',
                        opts=dict(
                            title=f"Original | Decoded",
                        ))
