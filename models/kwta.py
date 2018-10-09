import math

import torch
import torch.nn as nn

from utils.constants import SPARSITY


def get_kwta_threshold(tensor: torch.FloatTensor, sparsity: float):
    unsqueeze_dim = [1] * (tensor.ndimension() - 1)
    tensor = tensor.view(tensor.shape[0], -1)
    k_active = math.ceil(sparsity * tensor.shape[1])
    x_sorted, argsort = tensor.sort(dim=1, descending=True)
    threshold = x_sorted[:, [k_active-1, k_active]].mean(dim=1)
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


class KWinnersTakeAll(nn.Module):

    def __init__(self, sparsity=SPARSITY, connect_lateral=False):
        """
        :param sparsity: how many bits leave active
        :param connect_lateral: use lateral weights to wire activated neurons?
        """
        super().__init__()
        assert 0. <= sparsity <= 1., "Sparsity should lie in (0, 1) interval"
        self.sparsity = sparsity
        self.connect_lateral = connect_lateral
        self.weight_lateral = None

    def forward_kwta(self, x, sparsity):
        x = _KWinnersTakeAllFunction.apply(x, sparsity)
        return x

    def forward_lateral(self, x_input):
        batch_size, embedding_size = x_input.shape
        sparsity_forward = self.sparsity
        if not self.training:
            sparsity_forward /= 2
        if self.weight_lateral is None:
            self.weight_lateral = torch.zeros(embedding_size, embedding_size, device=x_input.device)
            sparsity_forward = self.sparsity
        x_output = self.forward_kwta(x_input, sparsity_forward)
        mask_active = x_output > 0.5
        if self.training:
            for mask_active_sample in mask_active:
                self.weight_lateral[mask_active_sample.nonzero(), mask_active_sample] += 1
        else:
            k_active_lateral = math.ceil((self.sparsity - sparsity_forward) * embedding_size)
            x_lateral = x_output @ self.weight_lateral
            x_lateral[mask_active] = 0  # don't select already active neurons
            _, argsort = x_lateral.sort(dim=1, descending=True)
            arange = torch.arange(batch_size, device=x_output.device).unsqueeze_(dim=1)
            x_output[arange, argsort[:, :k_active_lateral]] = 1
        return x_output

    def forward(self, x):
        if self.connect_lateral and x.ndimension() == 2:
            # auto-association is only for fully connected
            return self.forward_lateral(x)
        else:
            return self.forward_kwta(x, self.sparsity)

    def clear_lateral(self):
        self.weight_lateral = None

    def extra_repr(self):
        return f'sparsity={self.sparsity}, connect_lateral={self.connect_lateral}'


class KWinnersTakeAllSoft(KWinnersTakeAll):

    def __init__(self, sparsity=SPARSITY, connect_lateral=False, hardness=1):
        """
        :param sparsity: how many bits leave active
        :param connect_lateral: use lateral weights to wire activated neurons?
        :param hardness: exponent power in sigmoid function;
                         the larger the hardness, the closer sigmoid to the true kwta distribution.
        """
        super().__init__(sparsity=sparsity, connect_lateral=connect_lateral)
        self.hardness = hardness

    def forward_kwta(self, x, sparsity):
        if self.training:
            threshold = get_kwta_threshold(x, sparsity)
            x_scaled = self.hardness * (x - threshold)
            return x_scaled.sigmoid()
        else:
            return super().forward_kwta(x, sparsity)

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
