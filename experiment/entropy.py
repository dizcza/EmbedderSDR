import numpy as np
import torch
import torch.utils.data
from mighty.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm


def dataset_entropy(dataset_cls=MNIST):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    loader = DataLoader(dataset_cls, normalize=normalize)
    labels_full = []
    for images, labels in tqdm(loader, desc=f"Calculating {dataset_cls.__name__} labels entropy"):
        labels_full.append(labels)
    labels_full = torch.cat(labels_full)
    labels_unique, labels_count = np.unique(labels_full, return_counts=True)
    label_appearance_proba = labels_count / labels_count.sum()
    entropy = np.sum(label_appearance_proba * np.log2(1 / label_appearance_proba))
    print(f"{dataset_cls.__name__} entropy: {entropy:.3f} bits")


if __name__ == '__main__':
    dataset_entropy()
