import math

import torch
import torch.nn as nn

from constants import SPARSITY


def get_kwta_threshold(tensor: torch.FloatTensor, k_active):
    x_sorted, argsort = tensor.sort(dim=1, descending=True)
    threshold = x_sorted[:, [k_active-1, k_active]].mean(dim=1)
    threshold.unsqueeze_(dim=1)
    return threshold


class _KWinnersTakeAllFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, sparsity: float):
        batch_size, embedding_size = tensor.shape
        k_active = math.ceil(sparsity * embedding_size)
        threshold = get_kwta_threshold(tensor, k_active=k_active)
        mask_active = tensor > threshold
        return mask_active.type(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class KWinnersTakeAll(nn.Module):

    def __init__(self, sparsity=SPARSITY):
        """
        :param sparsity: how many bits leave active
        """
        super().__init__()
        assert 0. <= sparsity <= 1., "Sparsity should lie in (0, 1) interval"
        self.sparsity = sparsity

    def forward(self, x):
        x = _KWinnersTakeAllFunction.apply(x, self.sparsity)
        return x

    def extra_repr(self):
        return f'sparsity={self.sparsity}'


class KWinnersTakeAllSoft(KWinnersTakeAll):

    def __init__(self, sparsity=SPARSITY, hardness=10):
        """
        :param sparsity: how many bits leave active
        :param hardness: exponent power in sigmoid function;
                         the larger the hardness, the closer sigmoid to the true kwta distribution.
        """
        super().__init__(sparsity)
        self.hardness = hardness

    def forward(self, x):
        if self.training:
            batch_size, embedding_size = x.shape
            k_active = math.ceil(self.sparsity * embedding_size)
            threshold = get_kwta_threshold(x, k_active=k_active)
            x_scaled = self.hardness * (x - threshold)
            return x_scaled.sigmoid()
        else:
            return _KWinnersTakeAllFunction.apply(x, self.sparsity)

    def extra_repr(self):
        old_repr = super().extra_repr()
        return f"{old_repr}, hardness={self.hardness}"


class SynapticScaling(nn.Module):
    """
    Wrapper for KWTA to account synaptic scaling plasticity.
    """

    def __init__(self, kwta_layer: KWinnersTakeAll, synaptic_scale=1.0):
        super().__init__()
        self.kwta = kwta_layer
        self.synaptic_scale = synaptic_scale
        self.frequency = None
        self.seen = 0

    def forward(self, x):
        batch_size, embedding_size = x.shape
        if self.frequency is None:
            self.frequency = torch.zeros(embedding_size, device=x.device)
        scale = torch.exp(-self.synaptic_scale * self.frequency.clone())
        x = x * scale
        x = self.kwta(x)
        self.seen += batch_size
        self.frequency += (x.detach().sum(dim=0) - self.frequency * batch_size) / self.seen
        return x

    def extra_repr(self):
        return f"synaptic_scale={self.synaptic_scale}"
