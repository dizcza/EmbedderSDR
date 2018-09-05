import math
from abc import ABC, abstractmethod
from collections import defaultdict, namedtuple
from functools import wraps
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn import cluster
from sklearn.metrics import mutual_info_score

from monitor.batch_timer import Schedule
from monitor.viz import VisdomMighty

LayerForward = namedtuple("LayerForward", ("layer", "forward_orig"))
Information = namedtuple("Information", ("x", "y_percentiles"))


class MutualInfoBin(ABC):
    log2e = math.log2(math.e)
    n_bins_default = 20

    def __init__(self, estimate_size: int = np.inf, compression_range=(0.50, 0.999), n_percentiles=1, n_trials=1,
                 debug=False):
        """
        :param estimate_size: number of samples to estimate MI from
        :param compression_range: min & max acceptable quantization compression range
        :param n_percentiles: number of percentiles to divide the feature space into
        :param n_trials: number of trials to smooth the results for each percentile
        :param debug: plot bins distribution?
        """
        self.estimate_size = estimate_size
        self.compression_range = compression_range
        self.n_percentiles = n_percentiles
        self.n_trials = n_trials
        self.debug = debug
        self.layers_order = []
        self.n_bins = {}
        self.compression = {}
        self.max_trials_adjust = 10
        self.layers = {}
        self.activations = defaultdict(list)
        self.quantized = {}
        self.information = {}
        self.is_active = False
        self.eval_loader = None

    @property
    def percentiles(self):
        return 1. / 2 ** np.arange(start=self.n_percentiles - 1, stop=-1, step=-1, dtype=int)

    def register(self, layer: nn.Module, name: str):
        self.layers[name] = LayerForward(layer, layer.forward)

    @Schedule(epoch_update=0, batch_update=5)
    def force_update(self, model: nn.Module):
        if self.eval_loader is None:
            return
        self.start_listening()
        use_cuda = torch.cuda.is_available()
        with torch.no_grad():
            for batch_id, (images, labels) in enumerate(iter(self.eval_loader)):
                if use_cuda:
                    images = images.cuda()
                model(images)  # outputs of each layer are saved implicitly
                if batch_id * self.eval_loader.batch_size >= self.estimate_size:
                    break
        self.finish_listening()

    def decorate_evaluation(self, get_outputs_old: Callable):
        @wraps(get_outputs_old)
        def get_outputs_wrapped(*args, **kwargs):
            self.start_listening()
            outputs = get_outputs_old(*args, **kwargs)
            self.finish_listening()
            return outputs

        print(f"Decorated '{get_outputs_old.__name__}' function to save layer activations for MI estimation")
        return get_outputs_wrapped

    def prepare(self, loader: torch.utils.data.DataLoader, model: nn.Module, monitor_layers_count=5):
        self.eval_loader = loader
        inputs = []
        targets = []
        for images, labels in iter(loader):
            inputs.append(images)
            targets.append(labels)
            if len(inputs) * loader.batch_size >= self.estimate_size:
                break
        image_sample = inputs[0][0].unsqueeze_(dim=0)
        self.process(layer_name='input', activations=inputs)
        self.process(layer_name='target', activations=targets)
        self.start_listening()
        with torch.no_grad():
            if torch.cuda.is_available():
                image_sample = image_sample.cuda()
            model(image_sample)
        self.activations.clear()  # clear saved activations
        self.finish_listening()
        last_layer_names = self.layers_order[-monitor_layers_count:]
        last_layers = {}
        for name in last_layer_names:
            last_layers[name] = self.layers[name]
        self.layers = last_layers
        print(f"Monitoring only these last layers for mutual information estimation: {last_layer_names}")

    def start_listening(self):
        for name, (layer, forward_orig) in self.layers.items():
            if layer.forward == forward_orig:
                layer.forward = self._wrap_forward(layer_name=name, forward_orig=forward_orig)
        self.is_active = True

    def finish_listening(self):
        if not self.is_active:
            return
        for name, (layer, forward_orig) in self.layers.items():
            layer.forward = forward_orig
        self.is_active = False
        for hname in tuple(self.activations.keys()):
            self.process(layer_name=hname, activations=self.activations[hname])

    def _wrap_forward(self, layer_name, forward_orig):
        def forward_and_save(input):
            assert self.is_active, "Did you forget to call MutualInfo.start_listening()?"
            output = forward_orig(input)
            self.save_activations(layer_name, output)
            return output

        return forward_and_save

    def save_activations(self, layer_name: str, tensor: torch.Tensor):
        self.activations[layer_name].append(tensor.cpu().clone())
        if layer_name not in self.layers_order:
            self.layers_order.append(layer_name)

    def process(self, layer_name: str, activations: List[torch.FloatTensor]):
        activations = torch.cat(activations, dim=0)
        size = min(len(activations), self.estimate_size)
        activations = activations[: size]
        if layer_name == 'target':
            assert isinstance(activations, (torch.LongTensor, torch.IntTensor))
            activations = activations.numpy()
            self.quantized[layer_name] = activations
            return
        activations = activations.view(activations.shape[0], -1)
        if layer_name not in self.n_bins:
            self.adjust_bins(layer_name, activations)
        if layer_name == 'input':
            quantized = self.quantize(activations, n_bins=self.n_bins[layer_name])
            self.quantized[layer_name] = quantized
            return
        n_features = activations.shape[1]
        n_features_percentiles = np.ceil(n_features * self.percentiles).astype(int)
        self.quantized[layer_name] = []
        info_y = []
        for n_features_partial in n_features_percentiles:
            quantized_percentile = []
            info_y_percentile = []
            for trial in range(self.n_trials):
                feature_idx = np.random.choice(n_features, size=n_features_partial, replace=False)
                quantized_percentile_trial = self.quantize(activations[:, feature_idx],
                                                           n_bins=self.n_bins[layer_name])
                quantized_percentile.append(quantized_percentile_trial)
                info_y_trial = self.compute_mutual_info(self.quantized['target'], quantized_percentile_trial)
                info_y_percentile.append(info_y_trial)
            self.quantized[layer_name].append(quantized_percentile)
            info_y_percentile = np.mean(info_y_percentile)
            info_y.append(info_y_percentile)
        quantized_100 = self.quantized[layer_name][-1][0]  # any trial of 100-percentile is fine
        info_x = self.compute_mutual_info(self.quantized['input'], quantized_100)
        self.information[layer_name] = Information(x=info_x, y_percentiles=info_y)

    @staticmethod
    def compute_mutual_info(x, y) -> float:
        return mutual_info_score(x, y) * MutualInfoBin.log2e

    def plot_quantized_hist(self, viz):
        """
        Plots quantized bins distribution.
        Ideally, we'd like the histogram to match a uniform distribution.
        """
        def _plot_bins(name, quantized_trials):
            counts = []
            for quantized_trial in quantized_trials:
                _, counts_trial = np.unique(quantized_trial, return_counts=True)
                n_empty_clusters = self.n_bins[name] - len(counts_trial)
                counts_trial = np.r_[counts_trial, np.zeros(n_empty_clusters, dtype=int)]
                counts.append(counts_trial)
            counts = np.sort(counts, axis=1).mean(axis=0)
            counts = counts[::-1]
            viz.bar(X=counts, win=f'{name} MI hist', opts=dict(
                xlabel='bin ID',
                ylabel='# activation codes',
                title=f'MI quantized histogram: {name}',
            ))
        for name in self.quantized.keys():
            if name == 'target':
                continue
            elif name == 'input':
                quantized_trials = [self.quantized[name]]
            else:
                quantized_percentile_100 = self.quantized[name][-1]
                quantized_trials = quantized_percentile_100
            _plot_bins(name, quantized_trials=quantized_trials)

    def plot_compression(self, viz):
        viz.bar(X=list(self.compression.values()), win=f'compression', opts=dict(
            rownames=list(self.compression.keys()),
            ylabel='compression',
            title=f'MI quantized compression',
        ))

    def plot_information_percentiles(self, viz):
        info_y_percentiles = []
        legend = []
        for hname, information in self.information.items():
            info_y_percentiles.append(information.y_percentiles)
            legend.append(hname)
        title = f'Partial Mutual information'
        viz.line(Y=np.vstack(info_y_percentiles).T, X=100 * self.percentiles, win=title, opts=dict(
            xlabel='% of chosen neurons (percentile)',
            ylabel='I(T, Y), bits',
            title=title,
            legend=legend,
        ))

    def plot_activations_hist(self, viz):
        for hname, activations in self.activations.items():
            title = f'Activations histogram: {hname}'
            activations = torch.cat(activations, dim=0)
            viz.histogram(activations.view(-1), win=title, opts=dict(
                xlabel='neuron value',
                ylabel='neuron counts',
                title=title,
            ))

    def plot(self, viz):
        assert not self.is_active, "Wait, not finished yet."
        if len(self.information) == 0:
            return
        if self.debug:
            self.plot_quantized_hist(viz)
            self.plot_compression(viz)
            self.plot_activations_hist(viz)
        if self.n_percentiles > 1:
            self.plot_information_percentiles(viz)
        legend = []
        ys = []
        xs = []
        for layer_name, information in self.information.items():
            info_y_percentile_100 = information.y_percentiles[-1]
            ys.append(info_y_percentile_100)
            xs.append(information.x)
            legend.append(layer_name)
        title = 'Mutual information plane'
        viz.line(Y=np.array([ys]), X=np.array([xs]), win=title, opts=dict(
            xlabel='I(X, T), bits',
            ylabel='I(T, Y), bits',
            title=title,
            legend=legend,
        ), update='append' if viz.win_exists(title) else None)
        self.information.clear()
        self.activations.clear()

    def adjust_bins(self, layer_name: str, activations: torch.FloatTensor):
        n_bins = self.n_bins_default
        compression_min, compression_max = self.compression_range
        for trial in range(self.max_trials_adjust):
            quantized = self.quantize(activations, n_bins)
            unique = np.unique(quantized, axis=0)
            compression = (len(activations) - len(unique)) / len(activations)
            if compression > compression_max:
                n_bins *= 2
            elif compression < compression_min:
                n_bins = max(2, int(n_bins / 2))
                if n_bins == 2:
                    break
            else:
                self.compression[layer_name] = compression
                break
        self.n_bins[layer_name] = n_bins

    @abstractmethod
    def digitize(self, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        pass

    def quantize(self, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        digitized = self.digitize(activations, n_bins=n_bins)
        unique, inverse = np.unique(digitized, return_inverse=True, axis=0)
        return inverse


class MutualInfoKMeans(MutualInfoBin):

    def digitize(self, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        model = cluster.MiniBatchKMeans(n_clusters=n_bins)
        labels = model.fit_predict(activations)
        return labels

    def quantize(self, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        return self.digitize(activations, n_bins=n_bins)
