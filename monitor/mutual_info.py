import math
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn import cluster
from sklearn.metrics import mutual_info_score

from monitor.batch_timer import ScheduleExp
from utils.constants import BATCH_SIZE


class LayersOrder:
    def __init__(self, model: nn.Module):
        self.hooks = []
        self.layers_ordered = []
        self.register_hooks(model)

    def register_hooks(self, model: nn.Module):
        children = tuple(model.children())
        if any(children):
            for layer in children:
                self.register_hooks(layer)
        else:
            handle = model.register_forward_pre_hook(self.append_layer)
            self.hooks.append(handle)

    def append_layer(self, layer, tensor_input):
        self.layers_ordered.append(layer)

    def get_layers_ordered(self):
        for handle in self.hooks:
            handle.remove()
        return tuple(self.layers_ordered)


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
        self.n_bins = {}
        self.compression = {}
        self.max_trials_adjust = 10
        self.activations = defaultdict(list)
        self.quantized = {}
        self.information = {}
        self.is_active = False
        self.eval_loader = None
        self.layer_to_name = {}

    def register(self, layer: nn.Module, name: str):
        self.layer_to_name[layer] = name

    @ScheduleExp()
    def force_update(self, model: nn.Module):
        if self.eval_loader is None:
            return
        self.start_listening()
        use_cuda = torch.cuda.is_available()
        with torch.no_grad():
            for images, labels in self.eval_batches():
                if use_cuda:
                    images = images.cuda()
                model(images)  # outputs of each layer are saved implicitly
        self.finish_listening()

    def decorate_evaluation(self, get_outputs_old: Callable):
        @wraps(get_outputs_old)
        def get_outputs_wrapped(*args, **kwargs):
            self.start_listening()
            outputs = get_outputs_old(*args, **kwargs)
            self.finish_listening()
            return outputs
        return get_outputs_wrapped

    def eval_batches(self):
        n_samples = 0
        for images, labels in iter(self.eval_loader):
            if n_samples > self.estimate_size:
                break
            n_samples += len(labels)
            yield images, labels

    def prepare_input(self):
        inputs = []
        targets = []
        for images, labels in self.eval_batches():
            inputs.append(images)
            targets.append(labels)
        self.save_quantized(layer_name='input', activations=inputs)
        self.save_quantized(layer_name='target', activations=targets)
        image_sample = inputs[0][:1]
        return image_sample

    def prepare(self, loader: torch.utils.data.DataLoader, model: nn.Module, monitor_layers_count=5):
        self.eval_loader = loader
        image_sample = self.prepare_input()

        layers_order = LayersOrder(model)
        if torch.cuda.is_available():
            image_sample = image_sample.cuda()
        with torch.no_grad():
            model(image_sample)

        layers_ordered = layers_order.get_layers_ordered()
        layers_ordered = list(layer for layer in layers_ordered if layer in self.layer_to_name)
        layers_ordered = layers_ordered[-monitor_layers_count:]

        for layer in layers_ordered:
            layer.register_forward_hook(self.save_activations)

        monitored_layer_names = list(self.layer_to_name[layer] for layer in layers_ordered)
        print(f"Monitoring only these last layers for mutual information estimation: {monitored_layer_names}")

    def start_listening(self):
        self.activations.clear()
        self.is_active = True

    def finish_listening(self):
        self.is_active = False
        for hname, activations in self.activations.items():
            self.save_quantized(layer_name=hname, activations=activations)
        self.save_mutual_info()

    def save_activations(self, module: nn.Module, tensor_input, tensor_output):
        if not self.is_active:
            return
        layer_name = self.layer_to_name[module]
        if sum(map(len, self.activations[layer_name])) > self.estimate_size:
            return
        tensor_output_clone = tensor_output.cpu()
        if tensor_output_clone is tensor_output:
            tensor_output_clone = tensor_output_clone.clone()
        self.activations[layer_name].append(tensor_output_clone)

    def save_quantized(self, layer_name: str, activations: List[torch.FloatTensor]):
        activations = torch.cat(activations, dim=0)
        if layer_name == 'target':
            assert isinstance(activations, (torch.LongTensor, torch.IntTensor))
            activations = activations.numpy()
            self.quantized[layer_name] = activations
            return
        activations = activations.flatten(start_dim=1)
        if layer_name not in self.n_bins:
            self.adjust_bins(layer_name, activations)
        if layer_name == 'input':
            quantized_input = self.quantize(activations, n_bins=self.n_bins[layer_name])
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
    def quantize(self, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        pass


class MutualInfoKMeans(MutualInfoBin):

    def quantize(self, activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        model = cluster.MiniBatchKMeans(n_clusters=n_bins, batch_size=BATCH_SIZE)
        labels = model.fit_predict(activations)
        return labels

    def prepare_input(self):
        """
        Partial update.
        """
        image_sample = None
        targets = []
        classifier = cluster.MiniBatchKMeans(n_clusters=self.n_bins_default,
                                             batch_size=BATCH_SIZE,
                                             compute_labels=False)
        for images, labels in self.eval_batches():
            image_sample = images[:1]
            images = images.flatten(start_dim=1)
            classifier.partial_fit(images, labels)
            targets.append(labels)
        labels_predicted = []
        for images, _ in self.eval_batches():
            images = images.flatten(start_dim=1)
            labels_predicted.append(classifier.predict(images))
        self.quantized['input'] = np.hstack(labels_predicted)
        self.save_quantized(layer_name='target', activations=targets)
        return image_sample
