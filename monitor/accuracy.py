from models.kwta import KWinnersTakeAllFunction
from mighty.monitor.accuracy import AccuracyEmbedding


class AccuracyEmbeddingKWTA(AccuracyEmbedding):

    def __init__(self, metric='cosine'):
        super().__init__(metric=metric)
        self.sparsity = 1.

    def save(self, outputs_train, labels_train):
        super().save(outputs_train=outputs_train, labels_train=labels_train)
        # leave only k prominent values
        self.centroids = KWinnersTakeAllFunction.apply(self.centroids, self.sparsity)

    def extra_repr(self):
        return super().extra_repr() + f", sparsity={self.sparsity:.3f}"
