import math

import torch
import torch.nn as nn
from torch.nn.init import kaiming_uniform_

from models.kwta import KWinnersTakeAll, WinnerTakeAll
from .solver import basis_pursuit_admm


class _MatchingPursuitLinear(nn.Module):
    def __init__(self, in_features, out_features, trainable=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.randn(out_features, in_features)
        self.weight /= self.weight.norm(p=2, dim=1).unsqueeze(dim=1)
        self.weight = nn.Parameter(self.weight, requires_grad=trainable)

    def extra_repr(self):
        return f"in_features={self.in_features}, " \
               f"out_features={self.out_features}"


class MatchingPursuit(_MatchingPursuitLinear):
    def __init__(self, in_features, out_features, lamb=0.2,
                 solver=basis_pursuit_admm):
        super().__init__(in_features, out_features, trainable=True)
        self.lambd = lamb
        self.solver = solver

    def forward(self, x, lambd=None):
        input_shape = x.shape
        if lambd is None:
            lambd = self.lambd
        x = x.flatten(start_dim=1)
        with torch.no_grad():
            w_norm = self.weight.norm(p=2, dim=1).unsqueeze(dim=1)
            self.weight.div_(w_norm)
            encoded = self.solver(A=self.weight.t(), b=x,
                                  lambd=lambd, max_iters=100)
        decoded = encoded.matmul(self.weight)
        return encoded, decoded.view(*input_shape)

    def extra_repr(self):
        return f"{super().extra_repr()}, lambd={self.lambd}, " \
               f"solver={self.solver.__name__}"


class BinaryMatchingPursuit(_MatchingPursuitLinear):
    def __init__(self, in_features, out_features,
                 kwta: KWinnersTakeAll):
        super().__init__(in_features, out_features)
        self.kwta = kwta

    def forward(self, x: torch.Tensor, sparsity=None):
        if sparsity is None:
            sparsity = self.kwta.sparsity
        wta = WinnerTakeAll()
        input_shape = x.shape
        x = x.detach().flatten(start_dim=1)
        nonzero = x.shape[1] - (x == 0).sum(dim=1)
        lambd = 3 * nonzero.max() * self.in_features
        xr = torch.zeros_like(x)
        encoded = torch.zeros(x.shape[0], self.out_features,
                              dtype=torch.float32, device=x.device)
        k_active = math.ceil(sparsity * self.out_features)
        for step in range(k_active):
            residual = (2 * x - xr).matmul(self.weight.t()) - lambd * encoded
            encoded += wta(residual)
            xr = self.kwta(encoded.matmul(self.weight), sparsity)
        return encoded, xr.view(*input_shape)

    def extra_repr(self):
        return f"{super().extra_repr()}, kwta={self.kwta}"


class LISTA(nn.Module):
    def __init__(self, solver_model: MatchingPursuit, steps=3):
        super().__init__()
        solver_model.requires_grad_(False)
        self.solver_model = solver_model
        self.steps = steps
        self.weight = self.solver_model.weight.clone()
        self.weight.requires_grad_(True)
        self.shrink = nn.Softshrink(lambd=self.solver_model.lambd)
        self.s_matrix = nn.Parameter(torch.empty(self.out_features,
                                                 self.out_features),
                                     requires_grad=True)
        kaiming_uniform_(self.s_matrix, a=math.sqrt(5))

    @property
    def out_features(self):
        return self.solver_model.out_features

    @property
    def in_features(self):
        return self.solver_model.in_features

    def forward(self, x: torch.Tensor):
        bmp_encoded, bmp_decoded = self.solver_model(x)
        input_shape = x.shape
        x = x.flatten(start_dim=1)
        B = x.matmul(self.weight.t())  # (B, In) @ (In, V) -> (B, V)
        encoded = self.shrink(B)  # (B, V)
        for t in range(self.steps):
            C = B + encoded.matmul(self.s_matrix)  # (B, V)
            encoded = self.shrink(C)
        with torch.no_grad():
            decoded = encoded.matmul(self.weight).view(*input_shape)
        return encoded, decoded, bmp_encoded, bmp_decoded
