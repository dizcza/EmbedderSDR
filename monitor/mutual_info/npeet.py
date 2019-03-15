# Credits: Greg Ver Steeg

import warnings
from math import log
from typing import List

import numpy as np
import scipy.spatial as ss
import sklearn.decomposition
import torch
import torch.utils.data
import torch.utils.data
from scipy.special import digamma

from monitor.mutual_info._pca_preprocess import MutualInfoPCA


def entropy(x, k=3, base=2):
    """ The classic K-L k-nearest neighbor continuous entropy estimator
        x should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x = np.asarray(x)
    n_elements, n_features = x.shape
    x = add_noise(x)
    tree = ss.cKDTree(x)
    nn = query_neighbors(tree, x, k)
    const = digamma(n_elements) - digamma(k) + n_features * log(2)
    return (const + n_features * np.log(nn).mean()) / log(base)


def mi(x, y, z=None, k=3, base=2):
    """ Mutual information of x and y (conditioned on z if z is not None)
        x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
        if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x, y = np.asarray(x), np.asarray(y)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    if z is not None:
        points.append(z)
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = ss.cKDTree(points)
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = avgdigamma(x, dvec), avgdigamma(y, dvec), digamma(k), digamma(len(x))
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = avgdigamma(xz, dvec), avgdigamma(yz, dvec), avgdigamma(z, dvec), digamma(k)
    return (-a - b + c + d) / log(base)


def micd(x, y, k=3, base=2, warning=True):
    """ If x is continuous and y is discrete, compute mutual information
    """
    assert len(x) == len(y), "Arrays should have same length"
    entropy_x = entropy(x, k, base)

    y_unique, y_count = np.unique(y, return_counts=True, axis=0)
    y_proba = y_count / len(y)

    entropy_x_given_y = 0.
    for yval, py in zip(y_unique, y_proba):
        mask = y == yval
        if mask.ndim > 1:
            mask = mask.all(axis=1)
        x_given_y = x[mask]
        if k <= len(x_given_y) - 1:
            entropy_x_given_y += py * entropy(x_given_y, k, base)
        else:
            if warning:
                warnings.warn("Warning, after conditioning, on y={yval} insufficient data. "
                              "Assuming maximal entropy in this case.".format(yval=yval))
            entropy_x_given_y += py * entropy_x
    return abs(entropy_x - entropy_x_given_y)  # units already applied


def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)


def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1, p=float('inf'), n_jobs=-1)[0][:, k]


def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    n_elements = len(points)
    tree = ss.KDTree(points)
    avg = 0.
    dvec = dvec - 1e-15
    for point, dist in zip(points, dvec):
        # subtlety, we don't include the boundary point,
        # but we are implicitly adding 1 to kraskov def bc center point is included
        num_points = len(tree.query_ball_point(point, dist, p=float('inf')))
        avg += digamma(num_points) / n_elements
    return avg


class MutualInfoNPEET(MutualInfoPCA):
    """
    Kraskov mutual information estimation.
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
        self.quantized['input'] = (self.quantized['input'] -
                                   self.quantized['input'].mean()) / self.quantized['input'].std()
        for key in ['input', 'target']:
            self.quantized[key] = self.quantized[key].numpy()

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
            activations = (activations - activations.mean()) / activations.std()
            info_x = mi(self.quantized['input'], activations, k=4)
            info_y = micd(activations, self.quantized['target'], k=4)
            self.information[layer_name] = (info_x, info_y)
