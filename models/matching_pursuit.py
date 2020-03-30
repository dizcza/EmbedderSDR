import torch
import torch.nn as nn


def negligible_improvement(x, x_prev, tol):
    x_norm = x.norm(dim=1)
    dx_norm = (x - x_prev).norm(dim=1)
    return dx_norm / x_norm < tol


def basis_pursuit_admm(A, b, lambd, M_inv=None, tol=1e-4, max_iters=100):
    r"""
    PyTorch implementation of Basis Pursuit ADMM solver for the
    :math:`Q_1^\epsilon` problem.

    Parameters
    ----------
    A : (N, M) torch.Tensor
        The input weight matrix :math:`\boldsymbol{A}`.
    b : (B, N) torch.Tensor
        The right side of the equation :math:`\boldsymbol{A}\vec{x} = \vec{b}`.
    lambd : float
        :math:`\lambda`, controls the sparsity of :math:`\vec{x}`.
    tol : float
        The accuracy tolerance of ADMM.
    max_iters : int
        Run for at most `max_iters` iterations.

    Returns
    -------
    torch.Tensor
        (B, M) The solution vector batch :math:`\vec{x}`.
    """
    A_dot_b = b.matmul(A)
    if M_inv is None:
        M = A.t().matmul(A) + torch.eye(A.shape[1], device=A.device)
        M_inv = M.inverse().t_()
        del M

    batch_size = b.shape[0]
    v = torch.zeros(batch_size, A.shape[1], device=A.device)
    u = torch.zeros(batch_size, A.shape[1], device=A.device)
    v_prev = v.clone()

    v_solution = v.clone()
    solved = torch.zeros(batch_size, dtype=torch.bool)

    iter_id = 0
    for iter_id in range(max_iters):
        b_eff = A_dot_b + v - u
        x = b_eff.matmul(M_inv)  # M_inv is already transposed
        # x is of shape (<=B, m_atoms)
        v = torch.nn.functional.softshrink(x + u, lambd)
        u = u + x - v
        solved_batch = negligible_improvement(v, v_prev, tol=tol)
        if solved_batch.any():
            unsolved_ids = torch.nonzero(~solved)
            unsolved_ids.squeeze_(dim=1)
            keys = solved_batch.nonzero()
            keys.squeeze_(dim=1)
            became_solved_ids = unsolved_ids[keys]
            v_solution[became_solved_ids] = v[keys]
            solved[became_solved_ids] = True
            mask_unsolved = ~solved_batch
            v = v[mask_unsolved]
            u = u[mask_unsolved]
            A_dot_b = A_dot_b[mask_unsolved]
        if v.shape[0] == 0:
            # all solved
            break
        v_prev = v.clone()

    if iter_id != max_iters - 1:
        assert solved.all()
    # print(f"End iteration {iter_id}")

    return v_solution


class MatchingPursuit(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.randn(out_features, in_features)
        self.weight /= self.weight.norm(p=2, dim=0)
        A = self.weight.t()
        M = A.t().matmul(A) + torch.eye(A.shape[1], device=A.device)
        self.M_inv = M.inverse().t_()

    def forward(self, x, lambd=0.2):
        input_shape = x.shape
        x = x.flatten(start_dim=1)
        encoded = basis_pursuit_admm(A=self.weight.t(),
                                     b=x.detach(), M_inv=self.M_inv,
                                     lambd=lambd, max_iters=100)
        decoded = encoded.matmul(self.weight)
        return encoded, decoded.view(*input_shape)

    def extra_repr(self):
        return f"in_features={self.weight.shape[1]}, " \
               f"out_features={self.weight.shape[0]}"

    def cuda(self, device=None):
        super().cuda(device)
        self.weight = self.weight.cuda(device)
        self.M_inv = self.M_inv.cuda(device)
