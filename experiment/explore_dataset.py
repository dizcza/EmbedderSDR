import numpy as np
import torch
import torch.utils.data
import torch.utils.data
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from mighty.utils.var_online import VarianceOnline
from mighty.utils.constants import DATA_DIR
from mighty.utils.data import DataLoader


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


def check_normalized_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    dataset = MNIST(DATA_DIR, train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32,
                                         shuffle=False, num_workers=4)
    var_online = VarianceOnline()
    for images, labels in tqdm(
            loader,
            desc=f"Checking {MNIST.__name__} normalized transform"):
        for image in images:
            var_online.update(new_tensor=image)
    mean, std = var_online.get_mean_std()
    print(f"mean={mean.mean()}, std={std.mean()}")
