import os

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


def get_class_centroids(vectors: torch.FloatTensor, labels) -> torch.FloatTensor:
    assert len(vectors) == len(labels)
    centroids = []
    for label in labels.unique(sorted=True):
        centroids.append(vectors[labels == label].mean(dim=0))
    centroids = torch.stack(centroids, dim=0)
    return centroids


def calc_accuracy(labels_true, labels_predicted) -> float:
    accuracy = (labels_true == labels_predicted).type(torch.float32).mean()
    return accuracy.item()


def predict_centroid_labels(centroids: torch.FloatTensor, vectors_test: torch.FloatTensor):
    """
    Predicts the label based on L1 shortest distance to each of centroids.
    :param centroids: matrix of (n_classes, embedding_dim) shape
    :param vectors_test: matrix of (n_samples, embedding_dim) shape
    :return: predicted labels
    """
    distances = (vectors_test.unsqueeze(dim=1) - centroids).abs_().sum(dim=-1)
    labels_predicted = distances.argmin(dim=1)
    return labels_predicted


def argmax_accuracy(outputs, labels) -> float:
    labels_predicted = outputs.argmax(dim=1)
    return calc_accuracy(labels_predicted=labels_predicted, labels_true=labels)
