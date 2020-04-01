import math
import warnings

import torch
import torch.nn as nn

from mighty.monitor.var_online import MeanOnlineBatch
from utils.constants import SPARSITY
from utils.layers import SerializableModule


def get_kwta_threshold(tensor: torch.FloatTensor, sparsity: float):
    """
    Returns the threshold for kWTA activation function as if input tensor is a linear (batch x embedding_dim).

    :param tensor: (batch_size, embedding_dim) linear or (batch_size, c, w, h) conv tensor
    :param sparsity: kWTA sparsity
    :return: threshold for kWTA activation function to apply
    """
    unsqueeze_dim = [1] * (tensor.ndimension() - 1)
    tensor = tensor.flatten(start_dim=1)
    embedding_dim = tensor.shape[1]
    if embedding_dim < 2:
        raise ValueError(f"Embedding dimension {embedding_dim} should be >= 2")
    k_active = math.ceil(sparsity * embedding_dim)
    if k_active == embedding_dim:
        warnings.warn(f"kWTA cardinality {sparsity} is too high. "
                      f"Making 1 element equals zero.")
        k_active -= 1
    x_sorted, argsort = tensor.sort(dim=1, descending=True)
    threshold = x_sorted[:, [k_active - 1, k_active]].mean(dim=1)
    threshold = threshold.view(-1, *unsqueeze_dim)
    return threshold


class WinnerTakeAll(nn.Module):

    def forward(self, tensor: torch.Tensor):
        input_shape = tensor.shape
        tensor = tensor.flatten(start_dim=1)
        winners = tensor.max(dim=1).indices
        tensor.fill_(0.)
        tensor[range(tensor.shape[0]), winners] = 1.
        return tensor.view(*input_shape)


class KWinnersTakeAllFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, sparsity: float):
        if sparsity is None:
            return tensor
        threshold = get_kwta_threshold(tensor, sparsity)
        mask_active = tensor > threshold
        return mask_active.type(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class KWinnersTakeAll(nn.Module):
    """
    Non differentiable original k-winners-take-all activation function.
    It finds the top `k` units in a vector, sets them to one and the rest to zero.
    """

    def __init__(self, sparsity=SPARSITY):
        """
        :param sparsity: how many bits leave active
        """
        super().__init__()
        assert 0. < sparsity < 1., "Sparsity should lie in (0, 1) interval"
        self.register_buffer("sparsity", torch.tensor(float(sparsity),
                                                      dtype=torch.float32))

    def forward(self, x):
        x = KWinnersTakeAllFunction.apply(x, self.sparsity)
        return x

    def extra_repr(self):
        return f'sparsity={self.sparsity:.3f}'


class KWinnersTakeAllSoft(KWinnersTakeAll):
    """
    Differentiable version of k-winners-take-all activation function.
    Instead of a hard sign, it places the top `k` units of a vector on the right side of sigmod
      and the rest - on the left side of sigmoid.
    Hardness defines how well sigmoid resembles sign function.
    """

    def __init__(self, sparsity=SPARSITY, hardness=1):
        """
        :param sparsity: how many bits leave active
        :param hardness: exponent power in sigmoid function;
                         the larger the hardness, the closer sigmoid to the true kwta distribution.
        """
        super().__init__(sparsity=sparsity)
        self.register_buffer("hardness", torch.tensor(float(hardness),
                                                      dtype=torch.float32))

    def forward(self, x):
        if self.training:
            threshold = get_kwta_threshold(x, self.sparsity)
            x_scaled = self.hardness * (x - threshold)
            return x_scaled.sigmoid()
        else:
            return super().forward(x)

    def extra_repr(self):
        old_repr = super().extra_repr()
        return f"{old_repr}, hardness={self.hardness}"


class SynapticScaling(SerializableModule):
    """
    Wrapper for KWTA to account synaptic scaling plasticity.
    """

    state_attr = ['firing_rate']

    def __init__(self, kwta_layer: KWinnersTakeAll, synaptic_scale=1.0):
        super().__init__()
        self.kwta = kwta_layer
        self.register_buffer("synaptic_scale", torch.tensor(
            float(synaptic_scale), dtype=torch.float32))
        self.firing_rate = MeanOnlineBatch()

    @property
    def sparsity(self):
        return self.kwta.sparsity

    def forward(self, x):
        frequency = self.firing_rate.get_mean()
        if frequency is not None:
            scale = torch.exp(-self.synaptic_scale * frequency)
            x = x * scale
        x = self.kwta(x)
        self.firing_rate.update(x.detach())
        return x

    def extra_repr(self):
        return f"synaptic_scale={self.synaptic_scale:.3f}"
