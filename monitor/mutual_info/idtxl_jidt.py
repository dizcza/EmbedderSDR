from typing import List

import numpy as np
import sklearn
import torch
import torch.utils.data
from idtxl.estimators_jidt import JidtKraskovMI
from idtxl.estimators_opencl import OpenCLKraskovMI

from monitor.mutual_info._pca_preprocess import MutualInfoPCA


class MutualInfoIDTxl(MutualInfoPCA):

    def __init__(self, estimate_size=None, pca_size=100, debug=False):
        """
        :param estimate_size: number of samples to estimate mutual information from
        :param pca_size: transform input data to this size;
                               pass None to use original raw input data (no transformation is applied)
        :param debug: plot bins distribution?
        """
        super().__init__(estimate_size=estimate_size, pca_size=pca_size, debug=debug)
        settings = {'kraskov_k': 4}
        if torch.cuda.is_available():
            self.estimator = OpenCLKraskovMI(settings=settings)
        else:
            self.estimator = JidtKraskovMI(settings=settings)

    def prepare_input_finished(self):
        # self.quantized['input'] = (self.quantized['input'] -
        #                            self.quantized['input'].mean()) / self.quantized['input'].std()
        for key in ['input', 'target']:
            self.quantized[key] = self.quantized[key].numpy().astype(np.float64)

    def process_activations(self, layer_name: str, activations: List[torch.FloatTensor]):
        pass

    def save_mutual_info(self):
        hidden_layers_name = set(self.activations.keys())
        hidden_layers_name.difference_update({'input', 'target'})
        for layer_name in hidden_layers_name:
            activations = torch.cat(self.activations[layer_name]).numpy().astype(np.float64)
            if self.pca_size is not None and activations.shape[-1] > self.pca_size:
                pca = sklearn.decomposition.PCA(n_components=self.pca_size)
                activations = pca.fit_transform(activations)
            activations = (activations - activations.mean()) / activations.std()
            info_x = self.estimator.estimate(self.quantized['input'], activations)
            info_y = self.estimator.estimate(activations, self.quantized['target'])
            self.information[layer_name] = (info_x, info_y)
