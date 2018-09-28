import torch
import torch.utils.data
from torchvision import transforms, datasets
from tqdm import tqdm

from utils.constants import DATA_DIR
from monitor.viz import VisdomMighty


class VarianceOnline(object):

    """
    Online updating sample mean and unbiased variance in a single pass.
    """

    def __init__(self, tensor: torch.FloatTensor = None):
        self.mean = None
        self.var = None
        self.count = 0
        if tensor is not None:
            self.update(new_tensor=tensor)

    def update(self, new_tensor: torch.FloatTensor):
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


def dataset_mean_std(dataset_cls: type):
    """
    :param dataset_cls: class type of torch.utils.data.Dataset
    :return: samples' mean and std per channel, estimated from a training set
    """
    mean_std_file = DATA_DIR / "mean_std" / dataset_cls.__name__
    mean_std_file = mean_std_file.with_suffix('.pt')
    if not mean_std_file.exists():
        dataset = dataset_cls(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        var_online = VarianceOnline()
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
    Run as module: 'python -m monitor.var_online'.
    :param dataset_cls: class type of torch.utils.data.Dataset
    """
    viz = VisdomMighty(env="main")
    import time
    time.sleep(1)
    viz.close()
    mean, std = dataset_mean_std(dataset_cls=dataset_cls)
    viz.heatmap(mean.mean(dim=0), win=f'{dataset_cls.__name__} mean', opts=dict(
        title=f'{dataset_cls.__name__} Mean',
    ))
    viz.heatmap(std.mean(dim=0), win=f'{dataset_cls.__name__} std', opts=dict(
        title=f'{dataset_cls.__name__} STD',
    ))


if __name__ == '__main__':
    visualize_mean_std(dataset_cls=datasets.MNIST)
