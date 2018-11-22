import math
import random
from typing import List

import numpy as np
import sklearn.decomposition
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data
from tqdm import tqdm

from monitor.mutual_info.mutual_info import MutualInfo, AccuracyFromMutualInfo
from utils.algebra import onehot, exponential_moving_average
from utils.constants import BATCH_SIZE


class MutualInfoNeuralEstimationNetwork(nn.Module):
    """
    https://arxiv.org/pdf/1801.04062.pdf
    """

    hidden_units = (100, 50)  # number of hidden units in MINE net

    def __init__(self, x_size: int, y_size: int):
        """
        :param x_size: hidden layer shape
        :param y_size: input/target data shape
        """
        super().__init__()
        self.fc_x = nn.Linear(x_size, self.hidden_units[0])
        self.fc_y = nn.Linear(y_size, self.hidden_units[0])
        fc_hidden = []
        for in_features, out_features in zip(self.hidden_units[0:], self.hidden_units[1:]):
            fc_hidden.append(nn.Linear(in_features=in_features, out_features=out_features))
        self.fc_hidden = nn.Sequential(*fc_hidden)
        self.fc_output = nn.Linear(self.hidden_units[-1], 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):
        """
        :param x: some hidden layer batch activations of shape (batch_size, embedding_size)
        :param y: either input or target data samples of shape (batch_size, input_dimensions or 1)
        :return: mutual information I(x, y) approximation
        """
        hidden = self.relu(self.fc_x(x) + self.fc_y(y))
        hidden = self.relu(self.fc_hidden(hidden))
        output = self.fc_output(hidden)
        return output


class MutualInfoNeuralEstimationTrainer:
    filter_size = 30  # smoothing filter size
    filter_rounds = 3  # how many times apply
    learning_rate = 1e-4

    def __init__(self, mine_model: nn.Module):
        self.mine_model = mine_model
        self.optimizer = torch.optim.Adam(self.mine_model.parameters(), lr=self.learning_rate)
        self.mutual_info_history = [0]

    def __repr__(self):
        return f"{MutualInfoNeuralEstimationTrainer.__name__}(mine_model={self.mine_model}, optimizer={self.optimizer})"

    def start_training(self):
        self.mutual_info_history = [0]

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
        for repeat in range(self.filter_rounds):
            self.mutual_info_history = exponential_moving_average(self.mutual_info_history,
                                                                  window=self.filter_size)
        self.mutual_info_history *= MutualInfo.log2e  # convert nats to bits

    def get_mutual_info(self):
        return max(self.mutual_info_history)


class MutualInfoNeuralEstimation(MutualInfo):

    def __init__(self, estimate_size=float('inf'), estimate_epochs=3, input_pca_size=100, noise_variance=0.,
                 debug=False):
        """
        :param estimate_size: number of samples to estimate mutual information from
        :param estimate_epochs: total estimation epochs to run
        :param input_pca_size: transform input data to this size
        :param noise_variance: how much noise to add to input and targets
        :param debug: plot MINE training curves?
        """
        super().__init__(estimate_size=estimate_size, debug=debug)
        self.estimate_epochs = estimate_epochs
        self.noise_sampler = torch.distributions.normal.Normal(loc=0, scale=math.sqrt(noise_variance))
        self.trainers = {}  # MutualInformationNeuralEstimation trainers for both input X- and target Y-data
        self.input_size = input_pca_size
        self.target_size = None

    def extra_repr(self):
        return super().extra_repr() + f"; estimate_epochs={self.estimate_epochs}; input_pca_size={self.input_size}; " \
            f"noise_variance={self.noise_sampler.variance}; " \
            f"MINETrainer(filter_size={MutualInfoNeuralEstimationTrainer.filter_size}, " \
            f"filter_rounds={MutualInfoNeuralEstimationTrainer.filter_rounds}, " \
            f"optimizer.lr={MutualInfoNeuralEstimationTrainer.learning_rate}); " \
            f"MINE(hidden_units={MutualInfoNeuralEstimationNetwork.hidden_units})"

    def prepare_input(self):
        targets = []
        pca = sklearn.decomposition.IncrementalPCA(n_components=self.input_size, copy=False, batch_size=BATCH_SIZE)
        for images, labels in tqdm(self.eval_batches(), total=len(self.eval_loader),
                                   desc="MutualInfo: quantizing input data. Stage 1"):
            images = images.flatten(start_dim=1)
            pca.partial_fit(images, labels)
            targets.append(labels)
        targets = torch.cat(targets, dim=0)
        self.target_size = len(targets.unique())
        self.accuracy_estimator = AccuracyFromMutualInfo(n_classes=self.target_size)

        # one-hot encoded labels are better fit than argmax
        self.quantized['target'] = onehot(targets).type(torch.float32).split(self.eval_loader.batch_size)

        self.quantized['input'] = []
        for images, _ in tqdm(self.eval_batches(), total=len(self.eval_loader),
                              desc="MutualInfo: quantizing input data. Stage 2"):
            images = images.flatten(start_dim=1)
            images_transformed = pca.transform(images)
            images_transformed = torch.from_numpy(images_transformed).type(torch.float32)
            self.quantized['input'].append(images_transformed)

    def process_activations(self, layer_name: str, activations: List[torch.FloatTensor]):
        assert len(self.quantized['input']) == len(self.quantized['target']) == len(activations)
        embedding_size = activations[0].shape[1]
        if layer_name not in self.trainers:
            self.trainers[layer_name] = (
                MutualInfoNeuralEstimationTrainer(MutualInfoNeuralEstimationNetwork(x_size=embedding_size,
                                                                                    y_size=self.input_size)),
                MutualInfoNeuralEstimationTrainer(MutualInfoNeuralEstimationNetwork(x_size=embedding_size,
                                                                                    y_size=self.target_size)),
            )
        n_batches = len(activations)
        for mi_trainer in self.trainers[layer_name]:
            mi_trainer.start_training()
        for epoch in range(self.estimate_epochs):
            for batch_id in random.sample(range(n_batches), k=n_batches):
                for data_type, trainer in zip(('input', 'target'), self.trainers[layer_name]):
                    labels_batch = self.quantized[data_type][batch_id]
                    labels_batch = labels_batch + self.noise_sampler.sample(labels_batch.shape)
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
            info = np.transpose(info)
            title = f'MutualInfoNeuralEstimation {info_name}'
            viz.line(Y=info, X=np.arange(len(info)), win=title, opts=dict(
                xlabel='Iteration',
                ylabel='Mutual info lower bound, bits',
                title=title,
                legend=legend,
            ))

    def _plot_debug(self, viz):
        self.plot_activations_hist(viz)
        self.plot_mine_history_loss(viz)
