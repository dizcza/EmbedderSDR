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
from tqdm import tqdm

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


class MutualInfo(ABC):
    log2e = math.log2(math.e)
    n_bins_default = 20

    def __init__(self, estimate_size=float('inf'), debug=False):
        """
        :param estimate_size: number of samples to estimate mutual information from
        :param debug: plot bins distribution?
        """
        self.estimate_size = estimate_size
        self.debug = debug
        self.activations = defaultdict(list)
        self.quantized = {}
        self.information = {}
        self.is_active = False
        self.eval_loader = None
        self.layer_to_name = {}

    def __repr__(self):
        return f"{self.__class__.__name__}({self.extra_repr()})"

    def extra_repr(self):
        return f"estimate_size={self.estimate_size}"

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
        targets = []
        classifier = cluster.MiniBatchKMeans(n_clusters=self.n_bins_default,
                                             batch_size=BATCH_SIZE,
                                             compute_labels=False)
        for images, labels in tqdm(self.eval_batches(), total=len(self.eval_loader),
                                   desc="MutualInfo: quantizing input data. Stage 1"):
            images = images.flatten(start_dim=1)
            classifier.partial_fit(images, labels)
            targets.append(labels)
        targets = torch.cat(targets, dim=0)
        self.quantized['target'] = targets.numpy()

        centroids_predicted = []
        for images, _ in tqdm(self.eval_batches(), total=len(self.eval_loader),
                              desc="MutualInfo: quantizing input data. Stage 2"):
            images = images.flatten(start_dim=1)
            centroids_predicted.append(classifier.predict(images))
        self.quantized['input'] = np.hstack(centroids_predicted)

    def prepare(self, loader: torch.utils.data.DataLoader, model: nn.Module, monitor_layers_count=5):
        self.eval_loader = loader
        self.prepare_input()

        images_batch, _ = next(self.eval_batches())
        image_sample = images_batch[:1]
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
            self.process_activations(layer_name=hname, activations=activations)
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
        tensor_output_clone = tensor_output_clone.flatten(start_dim=1)
        self.activations[layer_name].append(tensor_output_clone)

    def plot_activations_hist(self, viz):
        for hname, activations in self.activations.items():
            title = f'Activations histogram: {hname}'
            activations = torch.cat(activations, dim=0)
            viz.histogram(activations.view(-1), win=title, opts=dict(
                xlabel='neuron value',
                ylabel='neuron counts',
                title=title,
            ))

    def _plot_debug(self, viz):
        self.plot_activations_hist(viz)

    def plot(self, viz):
        assert not self.is_active, "Wait, not finished yet."
        if len(self.information) == 0:
            return
        if self.debug:
            self._plot_debug(viz)
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

    @abstractmethod
    def process_activations(self, layer_name: str, activations: List[torch.FloatTensor]):
        pass

    @abstractmethod
    def save_mutual_info(self):
        pass

