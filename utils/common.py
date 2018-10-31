import math
import os
import time
from collections import namedtuple, defaultdict
from functools import lru_cache, wraps
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from torchvision import transforms, datasets

from utils.caltech import Caltech256, Caltech10
from utils.constants import DATA_DIR, MODELS_DIR, BATCH_SIZE
from utils.datasubset import MNIST56, FashionMNIST56, CIFAR10_56
from utils.normalize import NormalizeFromDataset

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
    func_duration = defaultdict(list)

    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        elapsed = time.time() - start
        elapsed *= 1e3
        func_duration[func.__name__].append(elapsed)
        print(f"{func.__name__} {elapsed: .3f} (mean: {np.mean(func_duration[func.__name__]): .3f}) ms")
        return res

    return wrapped


def get_data_loader(dataset: str, train=True, batch_size=BATCH_SIZE) -> torch.utils.data.DataLoader:
    if dataset == "MNIST56":
        dataset = MNIST56(train=train)
    elif dataset == "FashionMNIST56":
        dataset = FashionMNIST56(train=train)
    elif dataset == "CIFAR10_56":
        dataset = CIFAR10_56(train=train)
    elif dataset == "Caltech256":
        dataset = Caltech256(train=train)
    elif dataset == "Caltech10":
        dataset = Caltech10(train=train)
    else:
        if dataset == "MNIST":
            dataset_class = datasets.MNIST
        elif dataset == "FashionMNIST":
            dataset_class = datasets.FashionMNIST
        elif dataset == "CIFAR10":
            dataset_class = datasets.CIFAR10
        else:
            raise NotImplementedError()
        transform = transforms.Compose([transforms.ToTensor(), NormalizeFromDataset(dataset_cls=dataset_class)])
        dataset = dataset_class(DATA_DIR, train=train, download=True, transform=transform)
    num_workers = int(os.environ.get('LOADER_WORKERS', 4))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader


def load_model_state(dataset_name: str, model_name: str):
    model_path = MODELS_DIR.joinpath(dataset_name, Path(model_name).with_suffix('.pt'))
    if not model_path.exists():
        return None
    return torch.load(model_path)
