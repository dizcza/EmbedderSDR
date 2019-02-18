import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F


def compute_distance(input1, input2, metric, dim=1):
    if metric == 'cosine':
        dist = 1 - F.cosine_similarity(input1, input2, dim=dim)
    elif metric == 'l1':
        dist = F.l1_loss(input1, input2, reduction='none').sum(dim=dim)
    elif metric == 'l2':
        dist = F.mse_loss(input1, input2, reduction='none').sum(dim=dim)
    else:
        raise NotImplementedError
    return dist


def exponential_moving_average(array, window: int):
    array = np.asarray(array)
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = array.shape[0]

    pows = alpha_rev ** (np.arange(n + 1))

    scale_arr = 1 / pows[:-1]
    offset = array[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n - 1)

    mult = array * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


def onehot(y_labels):
    n_classes = len(y_labels.unique(sorted=False))
    y_onehot = torch.zeros(y_labels.shape[0], n_classes, dtype=torch.int64)
    y_onehot[torch.arange(y_onehot.shape[0]), y_labels] = 1
    return y_onehot


@lru_cache(maxsize=32, typed=False)
def factors_root(number: int):
    """
    :param number: an integer value
    :return: two integer factors, closest to the square root of the input
    """
    root = int(math.sqrt(number))
    for divisor in range(root, 0, -1):
        if number % divisor == 0:
            return divisor, number // divisor
    return 1, number


def pairwise_distances_matrix(input1, input2, metric='cosine'):
    r"""Compute the matrix of all squared pairwise distances.
    Arguments
    ---------
    input1 : torch.Tensor or Variable
        The first sample, should be of shape ``(n_1, d)``.
    input2 : torch.Tensor or Variable
        The second sample, should be of shape ``(n_2, d)``.
    metric : str
        The l_p norm to be used. Available: 'cosine', 'l1', 'l2'.
    Returns
    -------
    torch.Tensor or Variable
        Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
        ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
    input1 = input1.unsqueeze(1)
    input2 = input2.unsqueeze(0)
    if metric == 'cosine':
        dist = 1 - F.cosine_similarity(input1, input2, dim=2)
    elif metric == 'l1':
        dist = (input1 - input2).abs_().sum(dim=2)
    elif metric == 'l2':
        dist = (input1 - input2).pow_(2).sum(dim=2)
    else:
        raise NotImplementedError
    return dist
