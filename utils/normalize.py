import torch
from torchvision import transforms

from monitor.var_online import dataset_mean_std


class _NormalizeTensor:
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor.device)
        tensor = tensor.clone()
        tensor.sub_(mean).div_(std)
        return tensor


class NormalizeFromDataset(_NormalizeTensor):
    """
    Normalize dataset by subtracting channel-wise and pixel-wise mean and dividing by STD.
    Mean and STD are estimated from a training set only.
    """

    def __init__(self, dataset_cls: type):
        mean, std = dataset_mean_std(dataset_cls=dataset_cls)
        std += 1e-6
        super().__init__(mean=mean, std=std)


class NormalizeInverse(_NormalizeTensor):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def get_normalize_inverse(transform_composed: transforms.Compose):
    if transform_composed is None:
        return None
    for transform in transform_composed.transforms:
        if isinstance(transform, transforms.Normalize):
            return NormalizeInverse(mean=transform.mean, std=transform.std)
    return None
