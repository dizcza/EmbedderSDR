import math

import torch

from utils.constants import SPARSITY
from utils.layers import SerializableModule
from monitor.var_online import MeanOnlineBatch


def get_kwta_threshold(tensor: torch.FloatTensor, sparsity: float):
    unsqueeze_dim = [1] * (tensor.ndimension() - 1)
    tensor = tensor.view(tensor.shape[0], -1)
    k_active = math.ceil(sparsity * tensor.shape[1])
    x_sorted, argsort = tensor.sort(dim=1, descending=True)
    threshold = x_sorted[:, [k_active - 1, k_active]].mean(dim=1)
    threshold = threshold.view(-1, *unsqueeze_dim)
    return threshold


class _KWinnersTakeAllFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, sparsity: float):
        threshold = get_kwta_threshold(tensor, sparsity)
        mask_active = tensor > threshold
        return mask_active.type(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class KWinnersTakeAll(SerializableModule):

    state_attr = ["sparsity"]

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

    state_attr = KWinnersTakeAll.state_attr + ['hardness']

    def __init__(self, sparsity=SPARSITY, hardness=1):
        """
        :param sparsity: how many bits leave active
        :param hardness: exponent power in sigmoid function;
                         the larger the hardness, the closer sigmoid to the true kwta distribution.
        """
        super().__init__(sparsity=sparsity)
        self.hardness = hardness

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

    state_attr = ['synaptic_scale', 'frequency']

    def __init__(self, kwta_layer: KWinnersTakeAll, synaptic_scale=1.0):
        super().__init__()
        self.kwta = kwta_layer
        self.synaptic_scale = synaptic_scale
        self.frequency = MeanOnlineBatch()

    @property
    def sparsity(self):
        return self.kwta.sparsity

    def forward(self, x):
        frequency = self.frequency.get_mean()
        if frequency is not None:
            scale = torch.exp(-self.synaptic_scale * frequency)
            x = x * scale
        x = self.kwta(x)
        self.frequency.update(x.detach())
        return x

    def extra_repr(self):
        return f"synaptic_scale={self.synaptic_scale}"
