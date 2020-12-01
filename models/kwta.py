import math
import warnings
from typing import Union

import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F

from mighty.utils.var_online import MeanOnlineBatch
from utils.constants import SPARSITY
from mighty.models.serialize import SerializableModule


def compile_kwta(model: nn.Module):
    for name, child in model.named_modules():
        if isinstance(child, KWinnersTakeAllSoft):
            assert child.sparsity is not None, \
                "Only kWTA with fixed sparsity can be compiled"
            child_compiled = KWinnersTakeAll(sparsity=child.sparsity)
            setattr(model, name, child_compiled)
        compile_kwta(child)


class SparsityPredictor(nn.Module):
    def __init__(self, in_features: Union[int, None, str],
                 max_sparsity: float,
                 min_sparsity=0.001):
        super().__init__()
        assert min_sparsity < max_sparsity
        if in_features is None or in_features == "auto":
            self.linear = None
        else:
            self.linear = nn.Linear(in_features, out_features=1, bias=False)
        self.max_sparsity = max_sparsity
        self.min_sparsity = min_sparsity

    def forward(self, x):
        if self.linear is None:
            self.linear = nn.Linear(x.shape[1], out_features=1, bias=False)
            if torch.cuda.is_available():
                self.linear = self.linear.cuda()
        sparsity = self.linear(x).sigmoid() * self.max_sparsity
        return sparsity.squeeze()

    def extra_repr(self):
        extra = f"max_sparsity={self.max_sparsity}"
        if self.linear is None:
            extra = f"{extra}, linear='auto'"
        return extra

    def forward_threshold(self, x):
        assert x.ndimension() == 2, "Input tensor is assumed to be flattened"
        sparsity = self.forward(x)
        scale = x.std(dim=1)
        loc = x.mean(dim=1)
        gaussian = torch.distributions.Normal(loc, scale)
        sparsity = sparsity.clamp_min(min=self.min_sparsity)
        threshold = gaussian.icdf(1. - sparsity)
        return threshold


def get_kwta_threshold(tensor: torch.Tensor,
                       sparsity: Union[float, SparsityPredictor]):
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
    if isinstance(sparsity, SparsityPredictor):
        threshold = sparsity.forward_threshold(tensor)
    else:
        # float
        k_active = math.ceil(sparsity * embedding_dim)
        if k_active == embedding_dim:
            warnings.warn(f"kWTA cardinality {sparsity} is too high. "
                          f"Making 1 element equals zero.")
            k_active -= 1
        topk = tensor.topk(k_active + 1, dim=1).values
        threshold = topk[:, [-2, -1]].mean(dim=1)
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


class KWinnersTakeAllThresholdFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, threshold: torch.Tensor):
        if threshold is None:
            return tensor
        mask_active = tensor > threshold
        return mask_active.type(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class KWinnersTakeAllFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, sparsity: float):
        if sparsity is None:
            return tensor
        threshold = get_kwta_threshold(tensor, sparsity)
        return KWinnersTakeAllThresholdFunction.apply(tensor, threshold)

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
        self.sparsity = sparsity

    def forward(self, x: torch.Tensor):
        assert self.sparsity is not None, "Set the sparsity during the init"
        x = KWinnersTakeAllFunction.apply(x, self.sparsity)
        return x

    def extra_repr(self):
        if self.sparsity is None:
            return "sparsity='None'"
        elif isinstance(self.sparsity, float):
            return f"sparsity={self.sparsity:.3f}"
        return ''


class KWinnersTakeAllSoft(KWinnersTakeAll):
    """
    Differentiable version of k-winners-take-all activation function.
    Instead of a hard sign, it places the top `k` units of a vector on the right side of sigmod
      and the rest - on the left side of sigmoid.
    Hardness defines how well sigmoid resembles sign function.
    """

    def __init__(self, sparsity=None,
                 threshold_size: Union[int, None, str] = None,
                 hardness=1, hard=True):
        """
        :param sparsity: how many bits leave active
        :param hardness: exponent power in sigmoid function;
                         the larger the hardness, the closer sigmoid to the true kwta distribution.
        """
        if (sparsity is None and threshold_size is None) or \
                (sparsity is not None and threshold_size is not None):
            raise ValueError("Either 'sparsity' or 'threshold_size' "
                             "must be set, but not both.")
        super().__init__(sparsity=sparsity)
        self.threshold_size = threshold_size
        if self.sparsity is not None:
            self.threshold = None
        elif threshold_size is not None:
            self.sparsity = None
            if threshold_size == 'auto':
                self.threshold = 'auto'
            else:
                self.threshold = nn.Linear(in_features=threshold_size,
                                           out_features=1, bias=False).weight
        self.hardness = float(hardness)
        self.hard = hard

    def forward(self, x: torch.Tensor):
        if self.threshold is None:
            threshold = get_kwta_threshold(x, self.sparsity)
        else:
            if self.threshold == "auto":
                in_features = x.numel() // x.shape[0]
                self.threshold = nn.Linear(in_features,
                                           out_features=1,
                                           bias=False).weight
                if torch.cuda.is_available():
                    self.threshold = self.threshold.cuda()
            shape = list(x.shape)[1:]
            threshold = self.threshold.view(shape)
        if self.training:
            x_scaled = self.hardness * (x - threshold)
            if self.hard:
                return F.hardsigmoid(x_scaled, inplace=True)
            return x_scaled.sigmoid()
        return KWinnersTakeAllThresholdFunction.apply(x, threshold)

    def extra_repr(self):
        return f"{super().extra_repr()}, " \
               f"threshold_size={self.threshold_size}, " \
               f"hardness={self.hardness}".lstrip(', ')


class SynapticScaling(SerializableModule):
    """
    Wrapper for KWTA to account for synaptic scaling plasticity (also called
    the boost factor).
    """

    state_attr = ['firing_rate', 'boost_factor', 'target_sparsity']

    def __init__(self, kwta_layer: KWinnersTakeAll, boost_factor=1.0,
                 target_sparsity=0.):
        super().__init__()
        self.kwta = kwta_layer
        self.boost_factor = float(boost_factor)
        self.target_sparsity = target_sparsity
        self.firing_rate = MeanOnlineBatch()

    @property
    def sparsity(self):
        return self.kwta.sparsity

    def forward(self, x: torch.Tensor):
        if not self.training:
            # don't update firing rate on test
            return self.kwta(x)
        frequency = self.firing_rate.get_mean()
        if frequency is not None:
            logscale = self.boost_factor * (self.target_sparsity - frequency)
            x = x * torch.exp(logscale)
        x = self.kwta(x)
        self.firing_rate.update(x.detach())
        return x

    def extra_repr(self):
        return f"boost_factor={self.boost_factor:.3f}, " \
               f"target_sparsity={self.target_sparsity}"
