from torchvision.datasets import MNIST
from tqdm import tqdm

from mighty.monitor.var_online import MeanOnline
from mighty.utils.data import DataLoader


def dataset_sparsity(dataset_cls=MNIST, verbose=True):
    # L1 sparsity: ||x||_1 / size(x)
    #
    # MNIST:         0.131
    # FashionMNIST:  0.286
    # CIFAR10:       0.473
    # CIFAR100:      0.478
    loader = DataLoader(dataset_cls).get(train=True)
    sparsity_online = MeanOnline()
    for images, labels in tqdm(
            loader,
            desc=f"Computing {dataset_cls.__name__} sparsity",
            disable=not verbose,
            leave=False):
        images = images.flatten(start_dim=1)
        sparsity = images.norm(p=1, dim=1).mean() / images.shape[1]
        sparsity_online.update(sparsity)
    return sparsity_online.get_mean()


if __name__ == '__main__':
    sparsity = dataset_sparsity(MNIST)
    print(f"Input L1 sparsity: {sparsity:.3f}")
