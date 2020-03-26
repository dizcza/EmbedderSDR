from mighty.monitor.accuracy import AccuracyEmbedding

from models.kwta import KWinnersTakeAllFunction


class AccuracyEmbeddingKWTA(AccuracyEmbedding):

    def __init__(self, metric='cosine'):
        super().__init__(metric=metric)
        self.sparsity = 1.

    @property
    def centroids(self):
        centroids = super().centroids
        centroids = KWinnersTakeAllFunction.apply(centroids, self.sparsity)
        return centroids
