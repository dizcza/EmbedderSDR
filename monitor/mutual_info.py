import math
import random
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


class MutualInfoNeuralEstimationNetwork(nn.Module):
    """
    https://arxiv.org/pdf/1801.04062.pdf
    """

    n_hidden = 10  # number of hidden units in MINE net

    def __init__(self, x_size: int, y_size: int):
        """
        :param x_size: hidden layer shape
        :param y_size: input/target data shape
        """
        super().__init__()
        self.fc_x = nn.Linear(x_size, self.n_hidden)
        self.fc_y = nn.Linear(y_size, self.n_hidden)
        self.fc_output = nn.Linear(self.n_hidden, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        """
        :param x: some hidden layer batch activations of shape (batch_size, embedding_size)
        :param y: either input or target data samples of shape (batch_size, input_dimensions or 1)
        :return: mutual information I(x, y) approximation
        """
        hidden = self.fc_x(x) + self.fc_y(y)
        hidden = self.relu(hidden)
        output = self.fc_output(hidden)
        return output


class MINETrainer:
    flat_filter_size = 10  # smoothing filter size
    noise_variance = 0.2  # for smooth gradient flow

    def __init__(self, mine_model: nn.Module):
        self.mine_model = mine_model
        self.optimizer = torch.optim.Adam(self.mine_model.parameters(), lr=0.01)
        self.mutual_info_history = [0]
        self.noise_sampler = torch.distributions.normal.Normal(loc=0, scale=math.sqrt(self.noise_variance))

    def start_training(self):
        self.mutual_info_history = [0]

    def add_noise(self, activations):
        return activations + self.noise_sampler.sample(activations.shape)

    def train_batch(self, data_batch, labels_batch):
        self.optimizer.zero_grad()
        pred_joint = self.mine_model(data_batch, labels_batch)
        data_batch = data_batch[torch.randperm(data_batch.shape[0], device=data_batch.device)]
        labels_batch = labels_batch[torch.randperm(labels_batch.shape[0], device=labels_batch.device)]
        pred_marginal = self.mine_model(data_batch, labels_batch)
        mutual_info_lower_bound = torch.mean(pred_joint) - torch.log(torch.mean(torch.exp(pred_marginal)))
        self.mutual_info_history.append(mutual_info_lower_bound.item())
        loss = -mutual_info_lower_bound  # maximize
        loss.backward()
        self.optimizer.step()

    def finish_training(self):
        flat_filter = np.ones(self.flat_filter_size) / self.flat_filter_size
        self.mutual_info_history = np.convolve(flat_filter, self.mutual_info_history, mode='valid').tolist()

    def get_mutual_info(self):
        return self.mutual_info_history[-1]


class MutualInfo(ABC):
    log2e = math.log2(math.e)
    n_bins_default = 10

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
        image_sample = None
        targets = []
        classifier = cluster.MiniBatchKMeans(n_clusters=self.n_bins_default,
                                             batch_size=BATCH_SIZE,
                                             compute_labels=False)
        for images, labels in tqdm(self.eval_batches(), total=len(self.eval_loader),
                                   desc="MutualInfo: quantizing input data. Stage 1"):
            image_sample = images[:1]
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


class MutualInfoKMeans(MutualInfo):

    def __init__(self, estimate_size=float('inf'), debug=False, compression_range=(0.50, 0.999)):
        """
        :param estimate_size: number of samples to estimate mutual information from
        :param debug: plot bins distribution?
        :param compression_range: min & max acceptable quantization compression range
        """
        super().__init__(estimate_size=estimate_size, debug=debug)
        self.compression_range = compression_range
        self.n_bins = {}
        self.compression = {}
        self.max_trials_adjust = 10

    def process_activations(self, layer_name: str, activations: List[torch.FloatTensor]):
        activations = torch.cat(activations, dim=0)
        if layer_name not in self.n_bins:
            self.adjust_bins(layer_name, activations)
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
        return mutual_info_score(x, y) * MutualInfo.log2e

    @staticmethod
    def quantize(activations: torch.FloatTensor, n_bins: int) -> np.ndarray:
        model = cluster.MiniBatchKMeans(n_clusters=n_bins, batch_size=BATCH_SIZE, compute_labels=False)
        model.fit(activations)
        labels = model.predict(activations)
        return labels

    def adjust_bins(self, layer_name: str, activations: torch.FloatTensor):
        # todo use constant n_bins that equals total number of classes
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

    def _plot_debug(self, viz):
        super()._plot_debug(viz)
        self.plot_quantized_hist(viz)
        self.plot_compression(viz)


class MutualInfoNeuralEstimation(MutualInfo):

    def __init__(self, estimate_size=float('inf'), estimate_epochs=3, debug=False):
        """
        :param estimate_size: number of samples to estimate mutual information from
        :param estimate_epochs: total estimation epochs to run
        :param debug: plot bins distribution?
        """
        super().__init__(estimate_size=estimate_size, debug=debug)
        self.estimate_epochs = estimate_epochs
        self.trainers = {}  # MutualInformationNeuralEstimation trainers for both input X- and target Y-data

    def extra_repr(self):
        return super().extra_repr() + f", estimate_epochs={self.estimate_epochs}"

    def prepare_input(self):
        image_sample = super().prepare_input()
        for data_type in ('input', 'target'):
            tensor = torch.from_numpy(self.quantized[data_type])
            tensor = tensor.type(torch.float32)
            # not sure if normalization helps
            # tensor = (tensor - tensor.mean()) / tensor.std()
            tensor.unsqueeze_(dim=1)
            tensor = tensor.split(self.eval_loader.batch_size)
            self.quantized[data_type] = tensor
        return image_sample

    def process_activations(self, layer_name: str, activations: List[torch.FloatTensor]):
        assert len(self.quantized['input']) == len(self.quantized['target']) == len(activations)
        embedding_size = activations[0].shape[1]
        if layer_name not in self.trainers:
            mine_trainers = []
            for _unused in range(2):
                mine_model = MutualInfoNeuralEstimationNetwork(x_size=embedding_size, y_size=1)
                mine_trainer = MINETrainer(mine_model)
                mine_trainers.append(mine_trainer)
            self.trainers[layer_name] = tuple(mine_trainers)
        n_batches = len(activations)
        for mi_trainer in self.trainers[layer_name]:
            mi_trainer.start_training()
        for epoch in range(self.estimate_epochs):
            for batch_id in random.sample(range(n_batches), k=n_batches):
                for data_type, trainer in zip(('input', 'target'), self.trainers[layer_name]):
                    labels_batch = self.quantized[data_type][batch_id]
                    labels_batch = trainer.add_noise(labels_batch)
                    trainer.train_batch(data_batch=activations[batch_id], labels_batch=labels_batch)
        for mi_trainer in self.trainers[layer_name]:
            mi_trainer.finish_training()

    def save_mutual_info(self):
        for layer_name, (trainer_x, trainer_y) in self.trainers.items():
            info_x = trainer_x.get_mutual_info()
            info_y = trainer_y.get_mutual_info()
            self.information[layer_name] = (info_x, info_y)

    def plot_mine_history_loss(self, viz):
        legend = []
        info_x = []
        info_y = []
        for layer_name, (trainer_x, trainer_y) in self.trainers.items():
            info_x.append(trainer_x.mutual_info_history)
            info_y.append(trainer_y.mutual_info_history)
            legend.append(layer_name)
        for info_name, info in (('input X', info_x), ('target Y', info_y)):
            info = np.transpose(info) * self.log2e
            title = f'MutualInfoNeuralEstimation {info_name}'
            viz.line(Y=info, win=title, opts=dict(
                xlabel='Iteration',
                ylabel='Mutual info lower bound, bits',
                title=title,
                legend=legend,
            ))

    def _plot_debug(self, viz):
        self.plot_activations_hist(viz)
        self.plot_mine_history_loss(viz)
