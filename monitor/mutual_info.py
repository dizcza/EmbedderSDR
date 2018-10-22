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

from monitor.batch_timer import ScheduleExp

LayerForward = namedtuple("LayerForward", ("layer", "forward_orig"))


class MutualInfoBin(ABC):
    log2e = math.log2(math.e)
    n_bins_default = 20

    def __init__(self, estimate_size: int = np.inf, compression_range=(0.50, 0.999), debug=False):
        """
        :param estimate_size: number of samples to estimate MI from
        :param compression_range: min & max acceptable quantization compression range
        :param debug: plot bins distribution?
        """
        self.estimate_size = estimate_size
        self.compression_range = compression_range
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

    def register(self, layer: nn.Module, name: str):
        self.layers[name] = LayerForward(layer, layer.forward)

    @ScheduleExp()
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
        self.save_quantized(layer_name='input', activations=inputs)
        self.save_quantized(layer_name='target', activations=targets)
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
            self.save_quantized(layer_name=hname, activations=self.activations[hname])
        self.save_mutual_info()

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

    def save_quantized(self, layer_name: str, activations: List[torch.FloatTensor]):
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
            quantized_input = self.quantize_accurate(activations, n_bins=self.n_bins[layer_name])
            self.quantized[layer_name] = quantized_input
            return
        self.quantized[layer_name] = self.quantize(activations, n_bins=self.n_bins[layer_name])

    def save_mutual_info(self):
        hidden_layers_name = set(self.quantized.keys())
        hidden_layers_name.difference_update({'input', 'target'})
        for layer_name in hidden_layers_name:
            info_x = self.compute_mutual_info(self.quantized['input'], self.quantized[layer_name])
            info_y = self.compute_mutual_info(self.quantized['target'], self.quantized[layer_name])
            self.information[layer_name] = (info_x, info_y)

    @staticmethod
    def compute_mutual_info(x, y) -> float:
        return mutual_info_score(x, y) * MutualInfoBin.log2e

    def plot_quantized_hist(self, viz):
        """
        Plots quantized bins distribution.
        Ideally, we'd like the histogram to match a uniform distribution.
        """
        for layer_name in self.quantized.keys():
            if layer_name != 'target':
                _, counts = np.unique(self.quantized[layer_name], return_counts=True)
                n_empty_clusters = self.n_bins[layer_name] - len(counts)
                counts = np.r_[counts, np.zeros(n_empty_clusters, dtype=int)]
                counts.sort()
                counts = counts[::-1]
                title = f'MI quantized histogram: {layer_name}'
                viz.bar(X=counts, win=title, opts=dict(
                    xlabel='bin ID',
                    ylabel='# activation codes',
                    title=title,
                ))

    def plot_compression(self, viz):
        viz.bar(X=list(self.compression.values()), win=f'compression', opts=dict(
            rownames=list(self.compression.keys()),
            ylabel='compression',
            title=f'MI quantized compression',
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
        legend = []
        info_hidden_input = []
        info_hidden_output = []
        for layer_name, (info_x, info_y) in self.information.items():
            info_hidden_input.append(info_x)
            info_hidden_output.append(info_y)
            legend.append(layer_name)
        title = 'Mutual information plane'
        viz.line(Y=np.array([info_hidden_output]), X=np.array([info_hidden_input]), win=title, opts=dict(
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

    def quantize_accurate(self, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        # accurate version of quantize
        return self.quantize(activations=activations, n_bins=n_bins)


class MutualInfoKMeans(MutualInfoBin):

    def digitize(self, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        model = cluster.MiniBatchKMeans(n_clusters=n_bins)
        labels = model.fit_predict(activations)
        return labels

    def quantize(self, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        return self.digitize(activations, n_bins=n_bins)

    def quantize_accurate(self, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        model = cluster.KMeans(n_clusters=n_bins, n_jobs=-1)
        labels = model.fit_predict(activations)
        return labels
