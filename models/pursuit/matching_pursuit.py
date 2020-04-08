import math

import torch
import torch.nn as nn

from models.kwta import KWinnersTakeAll, WinnerTakeAll
from .solver import basis_pursuit_admm


class _MatchingPursuitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.randn(out_features, in_features)
        self.weight = nn.Parameter(self.weight, requires_grad=True)

    def normalize_weight(self):
        w_norm = self.weight.norm(p=2, dim=1).unsqueeze(dim=1)
        self.weight.div_(w_norm)

    def extra_repr(self):
        return f"in_features={self.in_features}, " \
               f"out_features={self.out_features}"


class MatchingPursuit(_MatchingPursuitLinear):
    def __init__(self, in_features, out_features, lamb=0.2,
                 solver=basis_pursuit_admm):
        super().__init__(in_features, out_features)
        self.lambd = lamb
        self.solver = solver

    def forward(self, x, lambd=None):
        input_shape = x.shape
        if lambd is None:
            lambd = self.lambd
        x = x.flatten(start_dim=1)
        with torch.no_grad():
            self.normalize_weight()
            encoded = self.solver(A=self.weight.t(), b=x,
                                  lambd=lambd, max_iters=100)
        decoded = encoded.matmul(self.weight)
        return encoded, decoded.view(*input_shape)

    def extra_repr(self):
        return f"{super().extra_repr()}, lambd={self.lambd}, " \
               f"solver={self.solver.__name__}"


class BinaryMatchingPursuit(_MatchingPursuitLinear):
    def __init__(self, in_features, out_features, kwta: KWinnersTakeAll):
        super().__init__(in_features, out_features)
        self.kwta = kwta

    @property
    def sparsity(self):
        return self.kwta.sparsity

    def forward(self, x: torch.Tensor):
        wta = WinnerTakeAll()
        input_shape = x.shape
        x = x.detach().flatten(start_dim=1)
        nonzero = x.shape[1] - (x == 0).sum(dim=1)
        lambd = 3 * nonzero.max() * self.in_features
        xr = torch.zeros_like(x)
        encoded = torch.zeros(x.shape[0], self.out_features,
                              dtype=torch.float32, device=x.device)
        k_active = math.ceil(self.sparsity * self.out_features)
        for step in range(k_active):
            residual = (2 * x - xr).matmul(self.weight.t()) - lambd * encoded
            encoded = encoded + wta(residual)
            xr = self.kwta(encoded.matmul(self.weight))
        return encoded, xr.view(*input_shape)

    def extra_repr(self):
        return f"{super().extra_repr()}, kwta={self.kwta}"
