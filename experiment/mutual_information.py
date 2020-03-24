import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from mighty.monitor.mutual_info.gcmi import micd
from mighty.monitor.mutual_info.kmeans import MutualInfoKMeans
from mighty.monitor.mutual_info.neural_estimation import MINE_Net, MINE_Trainer
from mighty.utils.common import set_seed
from mighty.utils.constants import BATCH_SIZE
from tqdm import trange


def synthesize_log_softmax_data(n_samples=50000, n_classes=10, p_argmax=0.95, onehot=False):
    """
    Simulates a typical trained neural network softmax output probabilities.

    Parameters
    ----------
    n_samples : int
        No. of samples
    n_classes : int
        No. of unique classes
    p_argmax : float
        Network accuracy (softmax probability of the correctly predicted class).
    onehot : bool
        Return as one-hot encoded vectors (True) or a list of class indices (False).

    Returns
    -------
    x_data : torch.FloatTensor
        Simulated softmax probabilities of shape (n_samples, n_classes)
    y_labels : torch.LongTensor
        Ground truth labels (class indices) of shape
            * (n_samples,), if `onehot` is False
            * (n_samples, n_classes), otherwise
    """
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
    trainer = MINE_Trainer(MINE_Net(x_size=outputs.shape[1], y_size=labels.shape[1]))

    outputs = outputs.split(BATCH_SIZE)
    labels = labels.split(BATCH_SIZE)
    n_batches = len(outputs)

    trainer.start_training()
    for epoch in trange(20, desc='Optimizing MINE'):
        for batch_id in random.sample(range(n_batches), k=n_batches):
            labels_batch = labels[batch_id]
            labels_batch += normal_sampler.sample(labels_batch.shape)
            trainer.train_batch(data_batch=outputs[batch_id], labels_batch=labels_batch)
    trainer.finish_training()
    print(f"Mutual Information Neural Estimation (MINE) lower-bound: {trainer.get_mutual_info():.3f}")
    plt.plot(np.arange(len(trainer.mutual_info_history)), trainer.mutual_info_history)
    plt.show()


def test_kmeans():
    outputs, labels = synthesize_log_softmax_data(n_samples=20000, n_classes=10, p_argmax=0.99)
    estimator = MutualInfoKMeans(n_bins=20, debug=False)
    quantized = estimator.quantize(outputs)
    estimated = MutualInfoKMeans.compute_mutual_info(quantized, labels)
    print(f"KMeans Mutual Information estimate: {estimated:.3f}")


def test_gcmi():
    """
    Test Gaussian-Copula Mutual Information estimator
    """
    outputs, labels = synthesize_log_softmax_data(n_samples=20000, n_classes=10, p_argmax=0.99)
    print(type(labels))
    estimated = micd(outputs.numpy().T, labels.numpy())
    print(f"Gaussian-Copula Mutual Information estimate: {estimated:.3f}")


if __name__ == '__main__':
    set_seed(26)
    # expected estimated value: log2(10) ~ 3.322
    test_kmeans()
    test_mine()
    test_gcmi()
