from torchvision.datasets import MNIST
from tqdm import tqdm

from mighty.monitor.var_online import MeanOnline
from mighty.utils.algebra import compute_sparsity
from mighty.utils.common import input_from_batch
from mighty.utils.data import DataLoader


def dataset_mean(data_loader: DataLoader, verbose=True):
    # L1 sparsity: ||x||_1 / size(x)
    #
    # MNIST:         0.131
    # FashionMNIST:  0.286
    # CIFAR10:       0.473
    # CIFAR100:      0.478
    loader = data_loader.get(train=True)
    sparsity_online = MeanOnline()
    for batch in tqdm(
            loader,
            desc=f"Computing {data_loader.dataset_cls.__name__} mean",
            disable=not verbose,
            leave=False):
        input = input_from_batch(batch)
        input = input.flatten(start_dim=1)
        sparsity = compute_sparsity(input)
        sparsity_online.update(sparsity)
    return sparsity_online.get_mean()


if __name__ == '__main__':
    sparsity = dataset_mean(DataLoader(MNIST))
    print(f"Input L1 sparsity: {sparsity:.3f}")
