from abc import ABC

import sklearn.decomposition
import torch
import torch.utils.data
import torch.utils.data
from tqdm import tqdm

from monitor.mutual_info.mutual_info import MutualInfo
from utils.constants import BATCH_SIZE


class MutualInfoPCA(MutualInfo, ABC):

    def __init__(self, estimate_size=None, pca_size=100, debug=False):
        """
        :param estimate_size: number of samples to estimate mutual information from
        :param pca_size: transform input data to this size;
                               pass None to use original raw input data (no transformation is applied)
        :param debug: plot bins distribution?
        """
        super().__init__(estimate_size=estimate_size, debug=debug)
        self.pca_size = pca_size

    def prepare_input_raw(self):
        inputs = []
        targets = []
        for images, labels in tqdm(self.eval_batches(), total=len(self.eval_loader),
                                   desc="MutualInfo: storing raw input data"):
            inputs.append(images.flatten(start_dim=1))
            targets.append(labels)
        self.quantized['input'] = torch.cat(inputs, dim=0)
        self.quantized['target'] = torch.cat(targets, dim=0)
        self.prepare_input_finished()

    def prepare_input_finished(self):
        pass

    def extra_repr(self):
        return super().extra_repr() + f"; pca_size={self.pca_size}"

    def prepare_input(self):
        if self.pca_size is None:
            self.prepare_input_raw()
            return
        images_batch, _ = next(iter(self.eval_loader))
        batch_size = images_batch.shape[0]
        assert batch_size >= self.pca_size, \
            f"Batch size {batch_size} has to be larger than PCA dim {self.pca_size} in order to run partial fit"
        targets = []
        pca = sklearn.decomposition.IncrementalPCA(n_components=self.pca_size, copy=False, batch_size=BATCH_SIZE)
        for images, labels in tqdm(self.eval_batches(), total=len(self.eval_loader),
                                   desc="MutualInfo: Applying PCA to input data. Stage 1"):
            if images.shape[0] < self.pca_size:
                # drop the last batch if it's too small
                continue
            images = images.flatten(start_dim=1)
            pca.partial_fit(images, labels)
            targets.append(labels)
        self.quantized['target'] = torch.cat(targets, dim=0)

        inputs = []
        for images, _ in tqdm(self.eval_batches(), total=len(self.eval_loader),
                              desc="MutualInfo: Applying PCA to input data. Stage 2"):
            images = images.flatten(start_dim=1)
            images_transformed = pca.transform(images)
            images_transformed = torch.from_numpy(images_transformed).type(torch.float32)
            inputs.append(images_transformed)
        self.quantized['input'] = torch.cat(inputs, dim=0)
        self.prepare_input_finished()
