"""
Gaussian copula mutual information estimation.

RAA Ince, BL Giordano, C Kayser, GA Rousselet, J Gross and PG Schyns,
A statistical framework for neuroimaging data analysis based on mutual
information estimated via a Gaussian copula.
Human Brain Mapping (2017) 38 p. 1541-1573 doi:10.1002/hbm.23471
"""

import warnings
from typing import List

import numpy as np
import scipy
import scipy.special
import sklearn.decomposition
import torch
import torch.utils.data
import torch.utils.data

from monitor.mutual_info._pca_preprocess import MutualInfoPCA


def ctransform(x):
    """Copula transformation (empirical CDF)

    cx = ctransform(x) returns the empirical CDF value along the first
    axis of x. Data is ranked and scaled within [0 1] (open interval).

    """
    xi = np.argsort(np.atleast_2d(x))
    xr = np.argsort(xi)
    cx = (xr + 1).astype(np.float) / (xr.shape[-1] + 1)
    return cx


def copnorm(x):
    """Copula normalization

    cx = copnorm(x) returns standard normal samples with the same empirical
    CDF value as the input. Operates along the last axis.

    """
    cx = scipy.special.ndtri(ctransform(x))
    return cx


def mi_gg(x, y, biascorrect=True, demeaned=False):
    """Mutual information (MI) between two Gaussian variables in bits

    I = mi_gg(x,y) returns the MI between two (possibly multidimensional)
    Gassian variables, x and y, with bias correction.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples last axis)

    :param x: (n_vars_x, n_trials)
    :param y: (n_vars_y, n_trials)
    :param biascorrect : true / false option (default true) which specifies whether
        bias correction should be applied to the esimtated MI.
    :param demeaned : false / true option (default false) which specifies whether th
        input data already has zero mean (true if it has been copula-normalized)

    :return mutual information in bits
    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarxy = Nvarx + Nvary

    if y.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # joint variable
    xy = np.vstack((x, y))
    if not demeaned:
        xy = xy - xy.mean(axis=1)[:, np.newaxis]
    Cxy = np.dot(xy, xy.T) / float(Ntrl - 1)
    # submatrices of joint covariance
    Cx = Cxy[:Nvarx, :Nvarx]
    Cy = Cxy[Nvarx:, Nvarx:]

    chCxy = np.linalg.cholesky(Cxy)
    chCx = np.linalg.cholesky(Cx)
    chCy = np.linalg.cholesky(Cy)

    # entropies in nats
    # normalizations cancel for mutual information
    HX = np.sum(np.log(np.diagonal(chCx)))  # + 0.5*Nvarx*(np.log(2*np.pi)+1.0)
    HY = np.sum(np.log(np.diagonal(chCy)))  # + 0.5*Nvary*(np.log(2*np.pi)+1.0)
    HXY = np.sum(
        np.log(np.diagonal(chCxy)))  # + 0.5*Nvarxy*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = scipy.special.psi(
            (Ntrl - np.arange(1, Nvarxy + 1, dtype=np.float32)) / 2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl - 1.0)) / 2.0
        HX = HX - Nvarx * dterm - psiterms[:Nvarx].sum()
        HY = HY - Nvary * dterm - psiterms[:Nvary].sum()
        HXY = HXY - Nvarxy * dterm - psiterms[:Nvarxy].sum()

    # MI in bits
    I = (HX + HY - HXY) / ln2
    return I


def gcmi_cc(x, y):
    """Gaussian-Copula Mutual Information between two continuous variables
    in bits.

    I = gcmi_cc(x,y) returns the MI between two (possibly multidimensional)
    continuous variables, x and y, estimated via a Gaussian copula.
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples first axis) 
    This provides a lower bound to the true MI value.

    :param x: (n_vars_x, n_trials)
    :param y: (n_vars_y, n_trials)
    :return mutual information in bits
    """

    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    if x.ndim > 2 or y.ndim > 2:
        raise ValueError("x and y must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]

    if y.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # check for repeated values
    for xi in range(Nvarx):
        if (np.unique(x[xi, :]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input x has more than 10% repeated values")
            break
    for yi in range(Nvary):
        if (np.unique(y[yi, :]).size / float(Ntrl)) < 0.9:
            warnings.warn("Input y has more than 10% repeated values")
            break

    # copula normalization
    cx = copnorm(x)
    cy = copnorm(y)
    # parametric Gaussian MI
    I = mi_gg(cx, cy, True, True)
    return I


def entropy_g(x, biascorrect=True):
    """Entropy of a Gaussian variable in bits

    H = ent_g(x) returns the entropy of a (possibly
    multidimensional) Gaussian variable x with bias correction.
    Columns of x correspond to samples, rows to dimensions/variables.
    (Samples last axis)

    """
    x = np.atleast_2d(x)
    if x.ndim > 2:
        raise ValueError("x must be at most 2d")
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]

    # demean data
    x = x - x.mean(axis=1)[:, np.newaxis]
    # covariance
    C = np.dot(x, x.T) / float(Ntrl - 1)
    chC = np.linalg.cholesky(C)

    # entropy in nats
    HX = np.sum(np.log(np.diagonal(chC))) + 0.5 * Nvarx * (
                np.log(2 * np.pi) + 1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = scipy.special.psi(
            (Ntrl - np.arange(1, Nvarx + 1, dtype=np.float32)) / 2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl - 1.0)) / 2.0
        HX = HX - Nvarx * dterm - psiterms.sum()

    # convert to bits
    return HX / ln2


def micd(x, y):
    """ If x is continuous and y is discrete, compute mutual information
    """
    assert x.shape[1] == len(y), "Arrays should have same length"
    entropy_x = entropy_g(x)

    y_unique, y_count = np.unique(y, return_counts=True, axis=0)
    y_proba = y_count / len(y)

    entropy_x_given_y = 0.
    for yval, py in zip(y_unique, y_proba):
        mask = y == yval
        x_given_y = x[:, mask]
        entropy_x_given_y += py * entropy_g(x_given_y)
    return abs(entropy_x - entropy_x_given_y)  # units already applied


class MutualInfoGCMI(MutualInfoPCA):
    """
    Gaussian-Copula Mutual Information estimation.
    """

    def __init__(self, estimate_size=None, pca_size=100, debug=False):
        """
        :param estimate_size: number of samples to estimate mutual information from
        :param pca_size: transform input data to this size;
                               pass None to use original raw input data (no transformation is applied)
        :param debug: plot MINE training curves?
        """
        super().__init__(estimate_size=estimate_size, pca_size=pca_size, debug=debug)

    def prepare_input_finished(self):
        for key in ['input', 'target']:
            self.quantized[key] = self.quantized[key].numpy().T

    def process_activations(self, layer_name: str, activations: List[torch.FloatTensor]):
        pass

    def save_mutual_info(self):
        hidden_layers_name = set(self.activations.keys())
        hidden_layers_name.difference_update({'input', 'target'})
        for layer_name in hidden_layers_name:
            activations = torch.cat(self.activations[layer_name]).numpy()
            if self.pca_size is not None and activations.shape[-1] > self.pca_size:
                pca = sklearn.decomposition.PCA(n_components=self.pca_size)
                activations = pca.fit_transform(activations)
            activations = activations.T
            info_x = gcmi_cc(self.quantized['input'], activations)
            info_y = micd(activations, self.quantized['target'])
            self.information[layer_name] = (info_x, info_y)
