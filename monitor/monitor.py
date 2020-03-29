import torch
import torch.utils.data
import torch.utils.data
import torch.utils.data

from mighty.monitor import Monitor
from mighty.monitor.batch_timer import ScheduleStep


class MonitorKWTA(Monitor):

    def update_sparsity(self, sparsity: float, mode: str):
        # L1 sparsity
        self.viz.line_update(y=sparsity, opts=dict(
            xlabel='Epoch',
            ylabel='L1 norm / size',
            title='Output sparsity',
        ), name=mode)

    def clusters_heatmap(self, mean, std):
        """
        Cluster centers distribution heatmap.

        Parameters
        ----------
        mean, std : torch.Tensor
            Tensors of shape `(C, V)`.
            The mean and standard deviation of `C` clusters (vectors of size
            `V`).

        """
        if mean.shape != std.shape:
            raise ValueError("The mean and std must have the same shape and"
                             "come from VarianceOnline.get_mean_std().")

        def compute_manhattan_dist(tensor: torch.FloatTensor) -> float:
            l1_dist = tensor.unsqueeze(dim=1) - tensor.unsqueeze(dim=0)
            l1_dist = l1_dist.norm(p=1, dim=2)
            upper_triangle_idx = l1_dist.triu_(1).nonzero(as_tuple=True)
            l1_dist = l1_dist[upper_triangle_idx].mean()
            return l1_dist

        n_classes = mean.shape[0]
        win = "Last layer activations heatmap"
        opts = dict(
            title=f"{win}. Epoch {self.timer.epoch}",
            xlabel='Embedding dimension',
            ylabel='Label',
            rownames=list(map(str, range(n_classes))),
        )
        if n_classes <= self.n_classes_format_ytickstep_1:
            opts.update(ytickstep=1)
        self.viz.heatmap(mean, win=win, opts=opts)
        self.save_heatmap(mean, win=win, opts=opts)
        normalizer = mean.norm(p=1, dim=1).mean()
        outer_distance = compute_manhattan_dist(mean) / normalizer
        std = std.norm(p=1, dim=1).mean() / normalizer
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

    def plot_autoencoder(self, images, reconstructed, reconstructed_binary,
                         n_show=10):
        assert images.shape == reconstructed.shape, "Input & decoded image shape differs"
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
            opts=dict(title="Original | Reconstructed | Reconstructed binary"))

    def plot_reconstruction_error(self, pixel_missed, thresholds):
        title = "Reconstruction error"
        self.viz.line(Y=pixel_missed, X=thresholds, win=title, opts=dict(
            title=title,
            xlabel="reconstruct threshold",
            ylabel="#incorrect_pixels"
        ))
        self.viz.line_update(pixel_missed.min(), opts=dict(
            title="Reconstruction error lowest",
            xlabel="Epoch",
            ylabel="min_thr #incorrect_pixels"
        ))
