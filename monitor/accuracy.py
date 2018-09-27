import os
from abc import ABC

import torch
import torch.nn as nn
import torch.utils.data


def how_many_samples_take(loader: torch.utils.data.DataLoader):
    """
    :param loader: train or test loader
    :return: number of samples to draw from
    """
    n_samples_take = -1
    is_train = getattr(loader.dataset, 'train', False)
    if is_train:
        # test dataset requires all samples
        n_samples_take = int(os.environ.get("FULL_FORWARD_PASS_SIZE", -1))
    if n_samples_take == -1:
        n_samples_take = float('inf')
    return n_samples_take


def get_outputs(model: nn.Module, loader: torch.utils.data.DataLoader):
    n_samples_take = how_many_samples_take(loader)
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []
    with torch.no_grad():
        for inputs, labels in iter(loader):
            if loader.batch_size * len(outputs_full) > n_samples_take:
                break
            if use_cuda:
                inputs = inputs.cuda()
            outputs = model(inputs)
            outputs_full.append(outputs.cpu())
            labels_full.append(labels)
    model.train(mode_saved)
    outputs_full = torch.cat(outputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    return outputs_full, labels_full


def calc_accuracy(labels_true, labels_predicted) -> float:
    accuracy = (labels_true == labels_predicted).type(torch.float32).mean()
    return accuracy.item()


class Accuracy(ABC):

    def save(self, outputs_train, labels_train):
        """
        If accuracy measure is not argmax (if the model doesn't end with a softmax layer),
        the output is embedding vector, which has to be stored and retrieved at prediction.
        :param outputs_train: model output on the train set
        :param labels_train: train set labels
        """
        pass

    def predict(self, outputs_test):
        """
        :param outputs_test: model output on the train or test set
        :return: predicted labels of shape (N,)
        """
        return self.predict_proba(outputs_test).argmax(dim=1)

    def predict_proba(self, outputs_test):
        """
        :param outputs_test: model output on the train or test set
        :return: predicted probabilities tensor of shape (N x C),
                 where C is the number of classes
        """
        raise NotImplementedError


class AccuracyArgmax(Accuracy):

    def predict(self, outputs_test):
        labels_predicted = outputs_test.argmax(dim=-1)
        return labels_predicted

    def predict_proba(self, outputs_test):
        return outputs_test.softmax(dim=1)


class AccuracyEmbedding(Accuracy):
    """
    Calculates the accuracy of embedding vectors.
    """

    def __init__(self):
        self.centroids = []

    def distances(self, outputs_test):
        assert len(self.centroids) > 0, "Save train embeddings first"
        centroids = torch.as_tensor(self.centroids, device=outputs_test.device)
        distances = (outputs_test.unsqueeze(dim=1) - centroids).abs_().sum(dim=-1)
        return distances

    def save(self, outputs_train, labels_train):
        outputs_train = outputs_train.detach()
        self.centroids = []
        for label in labels_train.unique(sorted=True):
            self.centroids.append(outputs_train[labels_train == label].mean(dim=0))
        self.centroids = torch.stack(self.centroids, dim=0).cpu()

    def predict(self, outputs_test):
        labels_predicted = self.distances(outputs_test).argmin(dim=1)
        return labels_predicted

    def predict_proba(self, outputs_test):
        distances = self.distances(outputs_test)
        proba = 1 - distances / distances.sum(dim=1).unsqueeze(1)
        return proba
