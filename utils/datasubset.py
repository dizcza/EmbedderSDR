from typing import Tuple

import torch
import torch.utils.data
from torchvision import transforms, datasets
from tqdm import tqdm

from utils.constants import DATA_DIR
from utils.normalize import NormalizeFromDataset


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
