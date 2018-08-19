import math
import time
from functools import lru_cache, wraps
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
from tqdm import tqdm

from constants import DATA_DIR, MODELS_DIR, MARGIN
from monitor.var_online import dataset_mean_std


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


def get_data_loader(dataset: str, train=True, batch_size=256) -> torch.utils.data.DataLoader:
    if dataset == "MNIST56":
        dataset = MNIST56(train=train)
    else:
        if dataset == "MNIST":
            dataset_cls = datasets.MNIST
        elif dataset == "CIFAR10":
            dataset_cls = datasets.CIFAR10
        else:
            raise NotImplementedError()
        transform = transforms.Compose([transforms.ToTensor(), NormalizeFromDataset(dataset_cls=dataset_cls)])
        dataset = dataset_cls(DATA_DIR, train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return loader


def load_model_state(dataset_name: str, model_name: str):
    model_path = MODELS_DIR.joinpath(dataset_name, Path(model_name).with_suffix('.pt'))
    if not model_path.exists():
        return None
    return torch.load(model_path)


def clamp_params(layer: nn.Module, a_min=0, a_max=1):
    for child in layer.children():
        clamp_params(child, a_min=a_min, a_max=a_max)
    if isinstance(layer, BinaryDecorator):
        for param in layer.parameters():
            param.data.clamp_(min=a_min, max=a_max)


class NormalizeFromDataset(transforms.Normalize):

    def __init__(self, dataset_cls: type):
        mean, std = dataset_mean_std(dataset_cls=dataset_cls)
        std += 1e-6
        super().__init__(mean=mean, std=std)


class ContrastiveLabeledLoss(nn.Module):
    """
    Even though this loss uses Euclidean distance, it's equivalent to l0 loss, since we apply KWinnersTakeAll.
    """

    def __init__(self, same_only=True):
        """
        :param same_only: use same-only or include same-other classes loss?
        """
        super().__init__()
        self.same_only = same_only

    def forward(self, outputs, labels):
        loss = 0
        for label_unique in labels.unique():
            outputs_same_label = outputs[labels == label_unique]
            if len(outputs_same_label) < 2:
                continue
            diff = outputs_same_label[1:] - outputs_same_label[0]
            euclidean_dist = torch.sum(torch.pow(diff, 2), dim=1)
            loss += euclidean_dist.mean()

            if not self.same_only:
                outputs_other_label = outputs[labels != label_unique]
                n_take = min(len(outputs_same_label), len(outputs_other_label))
                diff = outputs_other_label[:n_take] - outputs_same_label[:n_take]
                euclidean_dist = torch.sum(torch.pow(diff, 2), dim=1)
                euclidean_dist = torch.max(torch.zeros(len(euclidean_dist)), MARGIN - euclidean_dist)
                loss += euclidean_dist.mean()

        return loss


class MNIST56(torch.utils.data.TensorDataset):
    """
    MNIST 5 and 6 digits.
    """

    labels_keep = (5, 6)

    def __init__(self, train=True):
        self.train = train
        data_path = self.get_data_path()
        if not data_path.exists():
            mnist = datasets.MNIST(DATA_DIR, train=train, transform=transforms.ToTensor())
            self.process_mnist(mnist)
        with open(data_path, 'rb') as f:
            data, targets = torch.load(f)
        super().__init__(data, targets)

    def get_data_path(self):
        return DATA_DIR.joinpath(self.__class__.__name__, 'train.pt' if self.train else 'test.pt')

    def process_mnist(self, mnist: torch.utils.data.Dataset):
        data = []
        targets = []
        for image, label_old in tqdm(mnist, desc=f"Preparing {self.__class__.__name__} dataset"):
            if label_old in self.labels_keep:
                label_new = self.labels_keep.index(label_old)
                targets.append(label_new)
                data.append(image)
        data = torch.cat(data, dim=0)
        data_mean = data.mean(dim=0)
        data_std = data.std(dim=0) + 1e-6
        data = (data - data_mean) / data_std
        targets = torch.LongTensor(targets)
        data_path = self.get_data_path()
        data_path.parent.mkdir(exist_ok=True, parents=True)
        with open(data_path, 'wb') as f:
            torch.save((data, targets), f)
        print(f"Saved preprocessed data to {data_path}")


class BinaryDecorator(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        weight_full = self.layer.weight.data.clone()
        self.layer.weight.data = (weight_full > 0).type(torch.FloatTensor)
        x = self.layer(x)
        self.layer.weight.data = weight_full
        return x

    def __repr__(self):
        return "[Binary]" + repr(self.layer)
