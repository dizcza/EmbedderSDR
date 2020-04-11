from torchvision.datasets import MNIST
from tqdm import tqdm
import torch
from mighty.monitor.var_online import MeanOnline
from mighty.utils.data import DataLoader
from mighty.utils.algebra import compute_sparsity


def dataset_sparsity(dataset_cls=MNIST, verbose=True):
    # L1 sparsity: ||x||_1 / size(x)
    #
    # MNIST:         0.131
    # FashionMNIST:  0.286
    # CIFAR10:       0.473
    # CIFAR100:      0.478
    loader = DataLoader(dataset_cls).get(train=True)
    sparsity_online = MeanOnline()
    for batch in tqdm(
            loader,
            desc=f"Computing {dataset_cls.__name__} sparsity",
            disable=not verbose,
            leave=False):
        if isinstance(batch, torch.Tensor):
            input = batch
        else:
            input, labels = batch
        input = input.flatten(start_dim=1)
        sparsity = compute_sparsity(input)
        sparsity_online.update(sparsity)
    return sparsity_online.get_mean()


if __name__ == '__main__':
    sparsity = dataset_sparsity(MNIST)
    print(f"Input L1 sparsity: {sparsity:.3f}")
