import numpy as np
import torch
import torch.utils.data
from torchvision.datasets import MNIST
from tqdm import tqdm

from mighty.utils.data import DataLoader
from utils import dataset_sparsity


def dataset_entropy(dataset_cls=MNIST):
    # log2(10) = 3.322
    # log2(100) = 6.644
    #
    # MNIST:         3.320 bits
    # FashionMNIST:  3.322 bits
    # CIFAR10:       3.322 bits
    # CIFAR100:      6.644 bits
    loader = DataLoader(dataset_cls).get(train=True)
    labels_full = []
    for images, labels in tqdm(
            loader, desc=f"Computing {dataset_cls.__name__} labels entropy"):
        labels_full.append(labels)
    labels_full = torch.cat(labels_full)
    labels_unique, labels_count = np.unique(labels_full, return_counts=True)
    label_appearance_proba = labels_count / len(labels_full)
    entropy = np.sum(label_appearance_proba * np.log2(1 / label_appearance_proba))
    print(f"{dataset_cls.__name__} labels entropy: {entropy:.3f} bits")


if __name__ == '__main__':
    sparsity = dataset_sparsity(MNIST)
    print(f"Input L1 sparsity: {sparsity:.3f}")
