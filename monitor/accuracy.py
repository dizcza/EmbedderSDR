import torch

from mighty.monitor.accuracy import AccuracyEmbedding, AccuracyAutoencoder
from models.kwta import KWinnersTakeAllFunction


class AccuracyEmbeddingKWTA(AccuracyEmbedding):

    def __init__(self, metric='cosine', cache=False, sparsity=None):
        super().__init__(metric=metric, cache=cache)
        self.sparsity = sparsity

    @property
    def centroids(self):
        centroids = super().centroids
        centroids = KWinnersTakeAllFunction.apply(centroids, self.sparsity)
        return centroids

    def extra_repr(self):
        return f"{super().extra_repr()}, sparsity={self.sparsity}"


class AccuracyAutoencoderBinary(AccuracyAutoencoder, AccuracyEmbeddingKWTA):
    pass


class AccuracyEmbeddingLISTA(AccuracyEmbedding):
    def partial_fit(self, outputs_batch, labels_batch):
        encoded, decoded, bmp_encoded, bmp_decoded = outputs_batch
        super().partial_fit(encoded, labels_batch)

    def predict_cached(self):
        if not self.cache:
            raise ValueError("Caching is turned off")
        if len(self.input_cached) == 0:
            raise ValueError("Empty cached input buffer")
        input = torch.cat(self.input_cached,  dim=0)
        return super().predict(input)

    def predict(self, outputs_test):
        encoded, decoded, bmp_encoded, bmp_decoded = outputs_test
        return super().predict(encoded)
