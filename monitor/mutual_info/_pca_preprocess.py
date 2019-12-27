import pickle
from abc import ABC

import numpy as np
import sklearn.decomposition
import torch
import torch.utils.data
import torch.utils.data
from tqdm import tqdm

from monitor.mutual_info.mutual_info import MutualInfo
from utils.common import small_datasets
from utils.constants import BATCH_SIZE, PCA_DIR


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

        if self.eval_loader.dataset.__class__ in small_datasets():
            # for small datasets, use PCA on all images at once
            pca = self.pca_full()
        else:
            # otherwise, use incremental (batched) pca
            pca = self.pca_incremental()

        inputs = []
        targets = []
        for images, labels in tqdm(self.eval_batches(), total=len(self.eval_loader),
                                   desc="MutualInfo: Applying PCA to input data. Stage 2"):
            images = images.flatten(start_dim=1)
            images_transformed = pca.transform(images)
            images_transformed = torch.from_numpy(images_transformed).type(torch.float32)
            inputs.append(images_transformed)
            targets.append(labels)
        self.quantized['target'] = torch.cat(targets, dim=0)

        self.quantized['input'] = torch.cat(inputs, dim=0)
        self.prepare_input_finished()

    def pca_full(self):
        # memory inefficient
        dataset_name = self.eval_loader.dataset.__class__.__name__
        pca_path = PCA_DIR.joinpath(dataset_name, f"dim-{self.pca_size}.pkl")
        if not pca_path.exists():
            pca_path.parent.mkdir(parents=True, exist_ok=True)
            pca = sklearn.decomposition.PCA(n_components=self.pca_size, copy=False)
            images = np.vstack([im_batch.flatten(start_dim=1) for im_batch, _ in iter(self.eval_loader)])
            pca.fit(images)
            with open(pca_path, 'wb') as f:
                pickle.dump(pca, f)
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
        return pca

    def pca_incremental(self):
        pca = sklearn.decomposition.IncrementalPCA(n_components=self.pca_size, copy=False, batch_size=BATCH_SIZE)
        for images, _ in tqdm(self.eval_batches(), total=len(self.eval_loader),
                              desc="MutualInfo: Applying PCA to input data. Stage 1"):
            if images.shape[0] < self.pca_size:
                # drop the last batch if it's too small
                continue
            images = images.flatten(start_dim=1)
            pca.partial_fit(images)
        return pca
