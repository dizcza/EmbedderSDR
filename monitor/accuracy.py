from mighty.monitor.accuracy import AccuracyEmbedding
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
