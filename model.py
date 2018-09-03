import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import SPARSITY, EMBEDDING_SIZE


class BinarizeWeights(nn.Module):
    def __init__(self, layer: nn.Module):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        weight_full = self.layer.weight.data.clone()
        self.layer.weight.data = (weight_full > 0).type(torch.FloatTensor)
        x = self.layer(x)
        self.layer.weight.data = weight_full
        return x

    def __repr__(self):
        return "[Binary]" + repr(self.layer)


class _KWinnersTakeAllFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, sparsity: float):
        batch_size, embedding_size = tensor.shape
        _, argsort = tensor.sort(dim=1, descending=True)
        k_active = math.ceil(sparsity * embedding_size)
        active_indices = argsort[:, :k_active]
        mask_active = torch.ByteTensor(tensor.shape).zero_()
        mask_active[torch.arange(batch_size).unsqueeze_(dim=1), active_indices] = 1
        tensor[~mask_active] = 0
        tensor[mask_active] = 1
        # ctx.save_for_backward(mask_active)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class _KWinnersTakeAllFunctionSoft(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor, hardness, sparsity: float):
        batch_size, embedding_size = tensor.shape
        _, argsort = tensor.sort(dim=1, descending=True)
        k_active = math.ceil(sparsity * embedding_size)
        kth_element = tensor[torch.arange(batch_size), argsort[:, k_active]]
        kth_element.unsqueeze_(dim=1)
        tensor -= kth_element
        ctx.save_for_backward(tensor, hardness)
        tensor_scaled = hardness * tensor
        return tensor_scaled.sigmoid()

    @staticmethod
    def backward(ctx, grad_output):
        tensor, hardness = ctx.saved_tensors
        tensor_scaled = hardness * tensor
        tensor_exp = torch.exp(-tensor_scaled)
        grad_sigmoid = tensor_exp / torch.pow(1 + tensor_exp, 2)
        grad_output *= grad_sigmoid
        grad_input_tensor = grad_output * hardness
        grad_hardness = (grad_output * tensor).sum(dim=1).mean()
        return grad_input_tensor, grad_hardness, None


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
        self.hardness = nn.Parameter(torch.full((), hardness))

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


class Embedder(nn.Module):

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_emb(x)
        x = self.kwta(x)
        return x

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=5)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3)
        self.fc_emb = nn.Linear(in_features=5 * 8 * 8, out_features=EMBEDDING_SIZE, bias=False)
        self.kwta = KWinnersTakeAllSoft()
