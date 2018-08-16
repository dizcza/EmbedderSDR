import matplotlib.pyplot as plt
import torch
import torch.utils.data
from torchvision import transforms, datasets
from tqdm import tqdm

from constants import DATA_DIR


class VarianceOnline(object):

    """
    Online updating sample mean and unbiased variance in a single pass.
    """

    def __init__(self, tensor: torch.FloatTensor = None, is_active=False):
        self.mean = None
        self.var = None
        self.count = 0
        self.is_active = is_active
        if tensor is not None:
            self.update(new_tensor=tensor)

    def update(self, new_tensor: torch.FloatTensor):
        if not self.is_active:
            return
        self.count += 1
        if self.mean is None:
            self.mean = new_tensor.clone()
            self.var = torch.zeros_like(self.mean)
        else:
            self.var = (self.count - 2) / (self.count - 1) * self.var + torch.pow(new_tensor - self.mean, 2) / self.count
            self.mean += (new_tensor - self.mean) / self.count

    def get_mean_std(self):
        if self.mean is None:
            return None, None
        else:
            return self.mean.clone(), torch.sqrt(self.var)

    def reset(self):
        self.mean = None
        self.var = None
        self.count = 0

    def set_active(self, is_active: bool):
        if self.is_active and not is_active:
            # forget
            self.reset()
        self.is_active = is_active


def dataset_mean_std(dataset_cls: type, batch_size=256):
    """
    :param dataset_cls: class type of torch.utils.data.Dataset
    :param batch_size: batch size
    :return: samples' mean and std per channel, estimated from a training set
    """
    mean_std_file = DATA_DIR / "mean_std" / dataset_cls.__name__
    mean_std_file = mean_std_file.with_suffix('.pt')
    if not mean_std_file.exists():
        dataset = dataset_cls(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        var_online = VarianceOnline(is_active=True)
        for images, labels in tqdm(loader, desc=f"{dataset_cls.__name__}: running online mean, std"):
            for image in images:
                var_online.update(new_tensor=image)
        mean, std = var_online.get_mean_std()
        mean_std_file.parent.mkdir(exist_ok=True, parents=True)
        with open(mean_std_file, 'wb') as f:
            torch.save((mean, std), f)
    with open(mean_std_file, 'rb') as f:
        mean, std = torch.load(f)
    return mean, std


def visualize_mean_std(dataset_cls=datasets.MNIST):
    """
    Plots dataset mean and std, averaged across channels.
    :param dataset_cls: class type of torch.utils.data.Dataset
    """
    mean, std = dataset_mean_std(dataset_cls=dataset_cls)
    plt.subplot(121)
    plt.title(f"{dataset_cls.__name__} mean")
    plt.imshow(mean.mean(dim=0))
    plt.axis('off')
    plt.subplot(122)
    plt.title(f"{dataset_cls.__name__} STD")
    plt.imshow(std.mean(dim=0))
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    visualize_mean_std(dataset_cls=datasets.MNIST)
