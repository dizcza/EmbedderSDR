import numpy as np
import torch
from tqdm import tqdm

from utils.common import get_data_loader


def dataset_entropy(dataset_name="MNIST"):
    loader = get_data_loader(dataset=dataset_name)
    labels_full = []
    for images, labels in tqdm(loader, desc=f"Calculating {dataset_name} labels entropy"):
        labels_full.append(labels)
    labels_full = torch.cat(labels_full)
    labels_unique, labels_count = np.unique(labels_full, return_counts=True)
    label_appearance_proba = labels_count / labels_count.sum()
    entropy = np.sum(label_appearance_proba * np.log2(1 / label_appearance_proba))
    print(f"{dataset_name} entropy: {entropy:.3f} bits")


if __name__ == '__main__':
    dataset_entropy()
