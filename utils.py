import math
import time
from functools import lru_cache, wraps
from pathlib import Path
from typing import Tuple
from collections import namedtuple

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
from tqdm import tqdm

from constants import DATA_DIR, MODELS_DIR, BATCH_SIZE
from monitor.var_online import dataset_mean_std


DATASET_IMAGE_SIZE = {
    "MNIST": 28,
    "FashionMNIST": 28,
    "CIFAR10": 32,
}


AdversarialExamples = namedtuple("AdversarialExamples", ("original", "adversarial", "labels"))


@lru_cache(maxsize=32, typed=False)
def factors_root(number: int):
    """
    :param number: an integer value
    :return: two integer factors, closest to the square root of the input
    """
    root = int(math.sqrt(number))
    for divisor in range(root, 0, -1):
        if number % divisor == 0:
            return divisor, number // divisor
    return 1, number


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def timer_profile(func):
    """
    For debug purposes only.
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} {elapsed * 1e3} ms")
        return res

    return wrapped


def get_data_loader(dataset: str, train=True, batch_size=BATCH_SIZE) -> torch.utils.data.DataLoader:
    if dataset == "MNIST56":
        dataset = MNIST56(train=train)
    elif dataset == "FashionMNIST56":
        dataset = FashionMNIST56(train=train)
    elif dataset == "CIFAR10_56":
        dataset = CIFAR10_56(train=train)
    else:
        if dataset == "MNIST":
            dataset_class = datasets.MNIST
        elif dataset == "FashionMNIST":
            dataset_class = datasets.FashionMNIST
        elif dataset == "CIFAR10":
            dataset_class = datasets.CIFAR10
        else:
            raise NotImplementedError()
        transform = []
        if train:
            transform.append(transforms.RandomCrop(size=DATASET_IMAGE_SIZE[dataset], padding=4))
            transform.append(transforms.RandomHorizontalFlip())
        transform.extend([transforms.ToTensor(), NormalizeFromDataset(dataset_cls=dataset_class)])
        transform = transforms.Compose(transform)
        dataset = dataset_class(DATA_DIR, train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def load_model_state(dataset_name: str, model_name: str):
    model_path = MODELS_DIR.joinpath(dataset_name, Path(model_name).with_suffix('.pt'))
    if not model_path.exists():
        return None
    return torch.load(model_path)


def find_layers(model: nn.Module, layer_class):
    for name, layer in find_named_layers(model, layer_class=layer_class):
        yield layer


def find_named_layers(model: nn.Module, layer_class, name_prefix=''):
    for name, layer in model.named_children():
        yield from find_named_layers(layer, layer_class, name_prefix=f"{name_prefix}.{name}")
    if isinstance(model, layer_class):
        yield name_prefix.lstrip('.'), model


class NormalizeFromDataset(transforms.Normalize):
    """
    Normalize dataset by subtracting channel-wise and pixel-wise mean and dividing by STD.
    Mean and STD are estimated from a training set only.
    """

    def __init__(self, dataset_cls: type):
        mean, std = dataset_mean_std(dataset_cls=dataset_cls)
        std += 1e-6
        super().__init__(mean=mean, std=std)


class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)


def get_normalize_inverse(transform_composed: transforms.Compose):
    if transform_composed is None:
        return None
    for transform in transform_composed.transforms:
        if isinstance(transform, transforms.Normalize):
            return NormalizeInverse(mean=transform.mean, std=transform.std)
    return None


class DataSubset(torch.utils.data.TensorDataset):

    def __init__(self, dataset_cls, labels_keep: Tuple, train: bool):
        self.labels_keep = labels_keep
        self.train = train
        self.transform = transforms.Compose([transforms.ToTensor(), NormalizeFromDataset(dataset_cls=dataset_cls)])
        data_path = self.get_data_path()
        if not data_path.exists():
            original_dataset = dataset_cls(DATA_DIR, train=train, transform=self.transform, download=True)
            self.process_dataset(original_dataset)
        with open(data_path, 'rb') as f:
            data, targets = torch.load(f)
        super().__init__(data, targets)

    def get_data_path(self):
        return DATA_DIR.joinpath(self.__class__.__name__, 'train.pt' if self.train else 'test.pt')

    def process_dataset(self, dataset: torch.utils.data.Dataset):
        data = []
        targets = []
        train_str = "train" if self.train else "test"
        for image, label_old in tqdm(dataset, desc=f"Preparing {self.__class__.__name__} {train_str} dataset"):
            if label_old in self.labels_keep:
                label_new = self.labels_keep.index(label_old)
                targets.append(label_new)
                data.append(image)
        data = torch.stack(data, dim=0)
        targets = torch.LongTensor(targets)
        data_path = self.get_data_path()
        data_path.parent.mkdir(exist_ok=True, parents=True)
        with open(data_path, 'wb') as f:
            torch.save((data, targets), f)
        print(f"Saved preprocessed data to {data_path}")


class MNIST56(DataSubset):
    """
    MNIST 5 and 6 digits.
    """
    def __init__(self, train=True):
        super().__init__(dataset_cls=datasets.MNIST, labels_keep=(5, 6), train=train)


class FashionMNIST56(DataSubset):
    """
    FashionMNIST 5 and 6 classes.
    """
    def __init__(self, train=True):
        super().__init__(dataset_cls=datasets.FashionMNIST, labels_keep=(5, 6), train=train)


class CIFAR10_56(DataSubset):
    """
    CIFAR10 5 and 6 classes.
    """
    def __init__(self, train=True):
        super().__init__(dataset_cls=datasets.CIFAR10, labels_keep=(5, 6), train=train)
