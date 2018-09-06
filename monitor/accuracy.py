import psutil
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm


def get_outputs(model: nn.Module, loader: torch.utils.data.DataLoader):
    free_memory_limit = 1024 ** 3  # 1 Gb
    mode_saved = model.training
    model.train(False)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model.cuda()
    outputs_full = []
    labels_full = []
    with torch.no_grad():
        for inputs, labels in tqdm(iter(loader), desc="Full forward pass", leave=False):
            if psutil.virtual_memory().available < free_memory_limit:
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


def calc_accuracy(centroids: torch.FloatTensor, vectors_test: torch.FloatTensor, labels_test) -> float:
    """
    Calculates accuracy based on L1-measure distances between centroids of each class and vectors_test
    :param centroids: matrix of (n_classes, embedding_dim) shape
    :param vectors_test: matrix of (n_samples, embedding_dim) shape
    :param labels_test: true labels, n_samples
    :return: accuracy
    """
    distances = (vectors_test.unsqueeze(dim=1) - centroids).abs_().sum(dim=-1)
    labels_predicted = distances.argmin(dim=1)
    accuracy = (labels_test == labels_predicted).type(torch.FloatTensor).mean()
    return accuracy.item()


def calc_raw_accuracy(loader: torch.utils.data.DataLoader) -> float:
    inputs_full = []
    labels_full = []
    for inputs, labels in iter(loader):
        inputs_full.append(inputs)
        labels_full.append(labels)
    inputs_full = torch.cat(inputs_full, dim=0)
    labels_full = torch.cat(labels_full, dim=0)
    inputs_full = inputs_full.view(inputs_full.shape[0], -1)
    input_centroids = get_class_centroids(inputs_full, labels_full)
    accuracy_input = calc_accuracy(centroids=input_centroids, vectors_test=inputs_full,
                                   labels_test=labels_full)
    return accuracy_input
