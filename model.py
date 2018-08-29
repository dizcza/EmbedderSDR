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
        for sample_id in range(batch_size):
            mask_active[sample_id, active_indices[sample_id]] = 1
        tensor[~mask_active] = 0
        tensor[mask_active] = 1
        ctx.save_for_backward(mask_active)
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        mask_active, = ctx.saved_tensors
        return grad_output, None


class KWinnersTakeAll(nn.Module):

    def __init__(self, sparsity=SPARSITY):
        super().__init__()
        self.sparsity = sparsity

    def forward(self, x):
        x = _KWinnersTakeAllFunction.apply(x, self.sparsity)
        return x

    def extra_repr(self):
        return f'sparsity={self.sparsity}'


class Embedder(nn.Module):

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
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
        self.fc_emb = nn.Linear(in_features=5*8*8, out_features=EMBEDDING_SIZE, bias=False)
        self.kwta = KWinnersTakeAll()
