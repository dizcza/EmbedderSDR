import torch
from mighty.monitor.accuracy import AccuracyEmbedding

from models.kwta import KWinnersTakeAllFunction


class AccuracyEmbeddingKWTA(AccuracyEmbedding):

    def __init__(self, metric='cosine', cache=False):
        super().__init__(metric=metric, cache=cache)
        self.sparsity = 1.

    @property
    def centroids(self):
        centroids = super().centroids
        centroids = KWinnersTakeAllFunction.apply(centroids, self.sparsity)
        return centroids


class AccuracyEmbeddingAutoenc(AccuracyEmbeddingKWTA):
    def partial_fit(self, outputs_batch, labels_batch):
        latent, reconstructed = outputs_batch
        super().partial_fit(latent, labels_batch)

    def predict_cached(self):
        if not self.cache:
            raise ValueError("Caching is turned off")
        if len(self.input_cached) == 0:
            raise ValueError("Empty cached input buffer")
        input = torch.cat(self.input_cached,  dim=0)
        return super().predict(input)

    def predict(self, outputs_test):
        latent, reconstructed = outputs_test
        return super().predict(latent)
