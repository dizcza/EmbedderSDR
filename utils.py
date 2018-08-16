from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, datasets
from tqdm import tqdm

from constants import DATA_DIR, MODELS_DIR
from monitor.var_online import dataset_mean_std


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


class NormalizeFromDataset(transforms.Normalize):

    def __init__(self, dataset_cls: type):
        mean, std = dataset_mean_std(dataset_cls=dataset_cls)
        std += 1e-6
        super().__init__(mean=mean, std=std)


class ContrastiveLabeledLoss(nn.Module):

    def forward(self, outputs, labels):
        loss = torch.zeros(1)
        for label_unique in labels.unique():
            outputs_same_label = outputs[labels == label_unique]
            if len(outputs_same_label) < 2:
                continue
            diff = outputs_same_label[1:] - outputs_same_label[0]
            euclidean_dist = torch.sum(torch.pow(diff, 2), dim=1)
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
