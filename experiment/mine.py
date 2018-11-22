"""
Mutual Information Neural Estimation https://arxiv.org/pdf/1801.04062.pdf
"""

import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from monitor.mutual_info.kmeans import MutualInfoKMeans
from monitor.mutual_info.neural_estimation import MutualInfoNeuralEstimationNetwork, MutualInfoNeuralEstimationTrainer
from utils.common import set_seed
from utils.constants import BATCH_SIZE


def synthesize_log_softmax_data(n_samples=50000, n_classes=10, p_argmax=0.95, onehot=False):
    x_data = torch.randn(n_samples, n_classes)
    y_labels = x_data.argmax(dim=1)
    x_argmax = x_data[range(x_data.shape[0]), y_labels]
    softmax_sum = x_data.exp().sum(dim=1) - x_argmax
    x_argmax = torch.log(p_argmax * softmax_sum / (1 - p_argmax))
    x_data[range(x_data.shape[0]), y_labels] = x_argmax
    if onehot:
        y_onehot = torch.zeros(y_labels.shape[0], n_classes, dtype=torch.int64)
        y_onehot[torch.arange(y_onehot.shape[0]), y_labels] = 1
        y_labels = y_onehot
    return x_data, y_labels


def test_mine(var=0):
    outputs, labels = synthesize_log_softmax_data(n_samples=20000, n_classes=10, p_argmax=0.99, onehot=True)
    normal_sampler = torch.distributions.normal.Normal(loc=0, scale=math.sqrt(var))
    labels = labels.type(torch.float32)
    # outputs = (outputs - outputs.mean(dim=0)) / outputs.std(dim=0)
    trainer = MutualInfoNeuralEstimationTrainer(
        MutualInfoNeuralEstimationNetwork(x_size=outputs.shape[1], y_size=labels.shape[1]))

    outputs = outputs.split(BATCH_SIZE)
    labels = labels.split(BATCH_SIZE)
    n_batches = len(outputs)

    trainer.start_training()
    for epoch in range(20):
        for batch_id in tqdm(random.sample(range(n_batches), k=n_batches), desc=f'Epoch {epoch}'):
            labels_batch = labels[batch_id]
            labels_batch += normal_sampler.sample(labels_batch.shape)
            trainer.train_batch(data_batch=outputs[batch_id], labels_batch=labels_batch)
    trainer.finish_training()
    print(f"MINE mutual information lower-bound: {trainer.get_mutual_info()}")
    plt.plot(np.arange(len(trainer.mutual_info_history)), trainer.mutual_info_history)
    plt.show()


def test_kmeans():
    outputs, labels = synthesize_log_softmax_data(n_samples=20000, n_classes=10, p_argmax=0.99)
    mutual_info = MutualInfoKMeans(n_bins=20, debug=False)
    quantized = mutual_info.quantize(outputs)
    info = MutualInfoKMeans.compute_mutual_info(quantized, labels)
    print(f"KMeans mutual information estimate: {info}")


if __name__ == '__main__':
    set_seed(26)
    test_kmeans()
    test_mine()
