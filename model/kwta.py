import math

import torch
import torch.nn as nn

from constants import SPARSITY


class _KWinnersTakeAllFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, sparsity: float):
        batch_size, embedding_size = tensor.shape
        _, argsort = tensor.sort(dim=1, descending=True)
        k_active = math.ceil(sparsity * embedding_size)
        active_indices = argsort[:, :k_active]
        mask_active = torch.ByteTensor(tensor.shape).zero_()
        mask_active[torch.arange(batch_size).unsqueeze_(dim=1), active_indices] = 1
        mask_active[tensor == 0] = 0
        tensor[~mask_active] = 0
        tensor[mask_active] = 1
        # ctx.save_for_backward(mask_active)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class KWinnersTakeAll(nn.Module):

    def __init__(self, sparsity=SPARSITY):
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
        super().__init__(sparsity)
        self.hardness = hardness

    def forward(self, x):
        if self.training:
            batch_size, embedding_size = x.shape
            _, argsort = x.sort(dim=1, descending=True)
            k_active = math.ceil(self.sparsity * embedding_size)
            range_idx = torch.arange(batch_size)
            kth_element = x[range_idx, argsort[:, k_active]]
            if k_active < embedding_size:
                kth_next = x[range_idx, argsort[:, k_active+1]]
                threshold = (kth_element + kth_next) / 2
            else:
                threshold = kth_element
            threshold.unsqueeze_(dim=1)
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
