import math

import torch
import torch.nn as nn

from constants import SPARSITY


class _KWinnersTakeAllFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, sparsity: float, positive_only: bool):
        batch_size, embedding_size = tensor.shape
        _, argsort = tensor.sort(dim=1, descending=True)
        k_active = math.ceil(sparsity * embedding_size)
        active_indices = argsort[:, :k_active]
        mask_active = torch.ByteTensor(tensor.shape).zero_()
        range_idx = torch.arange(batch_size)

        mask_active[range_idx.unsqueeze(dim=1), active_indices] = 1
        if positive_only:
            mask_active[tensor <= 0] = 0
            # make sure at least one bit is on
            mask_active[range_idx, argsort[:, 0]] = 1

        tensor[~mask_active] = 0
        tensor[mask_active] = 1
        # ctx.save_for_backward(mask_active)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class KWinnersTakeAll(nn.Module):

    def __init__(self, sparsity=SPARSITY, with_relu=False):
        """
        :param sparsity: how many bits leave active
        :param with_relu: rely on positive only part of the signal?
        """
        super().__init__()
        assert 0. <= sparsity <= 1., "Sparsity should lie in (0, 1) interval"
        self.sparsity = sparsity
        if with_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    @property
    def with_relu(self):
        return self.relu is not None

    def forward(self, x):
        if self.with_relu:
            x = self.relu(x)
        x = _KWinnersTakeAllFunction.apply(x, self.sparsity, self.with_relu)
        return x

    def extra_repr(self):
        return f'sparsity={self.sparsity}, with_relu={self.with_relu}'


class KWinnersTakeAllSoft(KWinnersTakeAll):

    def __init__(self, sparsity=SPARSITY, with_relu=False, hardness=10):
        """
        :param sparsity: how many bits leave active
        :param with_relu: rely on positive only part of the signal?
        :param hardness: exponent power in sigmoid function;
                         the larger the hardness, the closer sigmoid to the true kwta distribution.
        """
        super().__init__(sparsity, with_relu=with_relu)
        self.hardness = hardness

    def forward(self, x):
        if self.with_relu:
            x = self.relu(x)
        if self.training:
            batch_size, embedding_size = x.shape
            _, argsort = x.sort(dim=1, descending=True)
            k_active = math.ceil(self.sparsity * embedding_size)
            kth_element_idx = argsort[:, k_active]
            if self.with_relu:
                last_positive = (x > 0).sum(dim=1) - 1
                last_positive.clamp_(min=0)
                kth_element_idx = torch.min(kth_element_idx, last_positive)
            kth_element_idx_next = (kth_element_idx + 1).clamp_(max=embedding_size-1)
            range_idx = torch.arange(batch_size)
            kth_element = x[range_idx, kth_element_idx]
            kth_next = x[range_idx, kth_element_idx_next]
            threshold = (kth_element + kth_next).unsqueeze_(dim=1) / 2
            x_scaled = self.hardness * (x - threshold)
            return x_scaled.sigmoid()
        else:
            return _KWinnersTakeAllFunction.apply(x, self.sparsity, self.with_relu)

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
