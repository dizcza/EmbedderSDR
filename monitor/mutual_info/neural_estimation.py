import math
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data

from monitor.mutual_info.mutual_info import MutualInfo


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


class MutualInfoNeuralEstimationTrainer:
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
        return max(self.mutual_info_history)


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
        super().prepare_input()
        for data_type in ('input', 'target'):
            tensor = torch.from_numpy(self.quantized[data_type])
            tensor = tensor.type(torch.float32)
            # not sure if normalization helps
            # tensor = (tensor - tensor.mean()) / tensor.std()
            tensor.unsqueeze_(dim=1)
            tensor = tensor.split(self.eval_loader.batch_size)
            self.quantized[data_type] = tensor

    def process_activations(self, layer_name: str, activations: List[torch.FloatTensor]):
        assert len(self.quantized['input']) == len(self.quantized['target']) == len(activations)
        embedding_size = activations[0].shape[1]
        if layer_name not in self.trainers:
            mine_trainers = []
            for _unused in range(2):
                mine_model = MutualInfoNeuralEstimationNetwork(x_size=embedding_size, y_size=1)
                mine_trainer = MutualInfoNeuralEstimationTrainer(mine_model)
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
