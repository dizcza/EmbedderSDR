import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import SPARSITY, EMBEDDING_SIZE


class KWinnersTakeAll(torch.autograd.Function):

    @staticmethod
    def forward(ctx, tensor):
        batch_size, embedding_size = tensor.shape
        _, argsort = tensor.sort(dim=1, descending=True)
        k_active = math.ceil(SPARSITY * embedding_size)
        active_indices = argsort[:, :k_active]
        mask_active = torch.ByteTensor(tensor.shape).zero_()
        for sample_id in range(batch_size):
            mask_active[sample_id, active_indices[sample_id]] = 1
        tensor[~mask_active] = 0
        tensor[mask_active] = 1
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Embedder(nn.Module):

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = self.conv1(x)
        x = self.bn1(x)
        F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=3)
        x = x.view(x.shape[0], -1)
        x = self.fc_emb(x)
        x = KWinnersTakeAll.apply(x)
        return x

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=5)
        self.fc_emb = nn.Linear(in_features=5*8*8, out_features=EMBEDDING_SIZE, bias=False)
