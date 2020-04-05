import torch
import torch.nn.functional as F


def negligible_improvement(x, x_prev, tol: float) -> torch.BoolTensor:
    x_norm = x.norm(dim=1)
    dx_norm = (x - x_prev).norm(dim=1)
    return dx_norm / x_norm < tol


def _reduce(solved, solved_batch, x_solution, x, *args):
    if solved_batch.any():
        args_reduced = []
        unsolved_ids = torch.nonzero(~solved)
        unsolved_ids.squeeze_(dim=1)
        keys = solved_batch.nonzero()
        keys.squeeze_(dim=1)
        became_solved_ids = unsolved_ids[keys]
        x_solution[became_solved_ids] = x[keys]
        solved[became_solved_ids] = True
        mask_unsolved = ~solved_batch
        x = x[mask_unsolved]
        args_reduced.append(x)
        for arg in args:
            arg = arg[mask_unsolved]
            args_reduced.append(arg)
    else:
        args_reduced = [x, *args]
    return args_reduced


def basis_pursuit_admm(A, b, lambd, M_inv=None, tol=1e-4, max_iters=100):
    r"""
    Basis Pursuit solver for the :math:`Q_1^\epsilon` problem

    .. math::
        \min_x \frac{1}{2} \left|\left| \boldsymbol{A}\vec{x} - \vec{b}
        \right|\right|_2^2 + \lambda \|x\|_1

    via the alternating direction method of multipliers (ADMM).

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
        M_inv = M.inverse().t()
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
        v = F.softshrink(x + u, lambd)
        u = u + x - v
        solved_batch = negligible_improvement(v, v_prev, tol=tol)
        v, u, A_dot_b = _reduce(solved, solved_batch, v_solution, v, u,
                                A_dot_b)
        if v.shape[0] == 0:
            # all solved
            break
        v_prev = v.clone()

    if iter_id != max_iters - 1:
        assert solved.all()
    v_solution[~solved] = v  # dump unsolved iterations

    return v_solution


def iterative_shrinkage(A, b, lambd, tol=1e-4, max_iters=100):
    r"""
    Iterative Shrinkage Algorithm (ISTA) for the :math:`Q_1^\epsilon` problem:

    .. math::
        \min_x \frac{1}{2} \left|\left| \boldsymbol{A}\vec{x} - \vec{b}
        \right|\right|_2^2 + \lambda \|x\|_1

    Parameters
    ----------
    A : (N, M) torch.Tensor
        The input weight matrix :math:`\boldsymbol{A}`.
    b : (B, N) torch.Tensor
        The right side of the equation :math:`\boldsymbol{A}\vec{x} = \vec{b}`.
    lambd : float
        :math:`\lambda`, controls the sparsity of :math:`\vec{x}`.
    tol : float
        The accuracy tolerance of ISTA.
    max_iters : int
        Run for at most `max_iters` iterations.

    Returns
    -------
    torch.Tensor
        (B, M) The solution vector batch :math:`\vec{x}`.
    """
    eigvals = torch.symeig(A.t().matmul(A)).eigenvalues
    # 1) abs() is technically not needed since the eigenvalues of a positive
    #    semi-definite symmetric matrix are non-negative
    # 2) multiplied by '2' because we need L > the-largest-eigval
    eigval_largest = 2 * eigvals.abs().max()
    lambd_norm = lambd / eigval_largest
    m_atoms = A.shape[1]
    # x is of shape (B, M)
    x = torch.zeros(b.shape[0], m_atoms, dtype=torch.float32, device=A.device)
    x_prev = x.clone()

    x_solution = x.clone()
    solved = torch.zeros(b.shape[0], dtype=torch.bool)
    iter_id = 0

    for iter_id in range(max_iters):
        # x_unconstrained is before applying soft shrinkage
        x_unconstrained = x - (x.matmul(A.t()) - b).matmul(A) / eigval_largest
        x = F.softshrink(x_unconstrained, lambd=lambd_norm)
        solved_batch = negligible_improvement(x, x_prev, tol=tol)
        x, b = _reduce(solved, solved_batch, x_solution, x, b)
        if x.shape[0] == 0:
            # all solved
            break
        x_prev = x.copy()

    if iter_id != max_iters - 1:
        assert solved.all()
    x_solution[~solved] = x  # dump unsolved iterations

    return x_solution
