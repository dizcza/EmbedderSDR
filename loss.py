import math
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monitor.batch_timer import timer
from utils.layers import SerializableModule


class ContrastiveLoss(nn.Module, ABC):

    def __init__(self, metric='cosine'):
        """
        :param metric: cosine, l2 or l1 metric to measure the distance between embeddings
        """
        super().__init__()
        self.metric = metric
        self.eps = 1e-6
        if self.metric == 'cosine':
            self.margin = 0.5
        else:
            self.margin = 0.75

    @property
    def power(self):
        if self.metric == 'l1':
            return 1
        elif self.metric == 'l2':
            return 2
        else:
            raise NotImplementedError

    def extra_repr(self):
        return f'metric={self.metric}, margin={self.margin}'

    @staticmethod
    def filter_nonzero(outputs, labels):
        nonzero = (outputs != 0).any(dim=1)
        return outputs[nonzero], labels[nonzero]

    @staticmethod
    def mean_nonempty(distances):
        return 0 if len(distances) == 0 else distances.mean()

    def distance(self, input1, input2):
        if self.metric == 'cosine':
            dist = 1 - F.cosine_similarity(input1, input2, dim=1)
        elif self.metric == 'l1':
            dist = (input1 - input2).abs().sum(dim=1)
        elif self.metric == 'l2':
            dist = (input1 - input2).pow(2).sum(dim=1)
        else:
            raise NotImplementedError
        return dist

    def forward_mean(self, outputs, labels):
        """
        :param outputs: (B, D) embeddings
        :param labels: (B,) labels
        :return: same-other loss on mean activations only
        """
        labels_unique = labels.unique(sorted=True).tolist()
        outputs_mean = []
        for label in labels_unique:
            outputs_mean.append(outputs[labels == label].mean(dim=0))
        outputs_mean = torch.stack(outputs_mean)
        loss = 0
        for label_id, label_same in enumerate(labels_unique[:-1]):
            outputs_same = outputs_mean[label_same].unsqueeze(dim=0)
            dist = self.distance(outputs_same, outputs_mean[label_id + 1:])
            loss = loss + dist.mean()
        return loss


class LossFixedPattern(ContrastiveLoss, SerializableModule):
    state_attr = ['patterns']

    def __init__(self, sparsity: float, metric='cosine'):
        super().__init__(metric=metric)
        self.patterns = {}
        self.sparsity = sparsity

    def forward(self, outputs, labels):
        outputs, labels = self.filter_nonzero(outputs, labels)
        embedding_dim = outputs.shape[1]
        n_active = math.ceil(self.sparsity * embedding_dim)
        loss = 0
        for label in labels.unique().tolist():
            if label not in self.patterns:
                code = torch.zeros(embedding_dim, device=outputs.device)
                code[:n_active] = 1
                code = code[torch.randperm(embedding_dim, device=code.device)]
                self.patterns[label] = code
            outputs_same_label = outputs[labels == label]
            pattern = torch.as_tensor(self.patterns[label], device=outputs_same_label.device)
            pattern = pattern.expand_as(outputs_same_label)
            dist = self.distance(outputs_same_label, pattern)
            loss = loss + dist.mean()

        return loss


class ContrastiveLossBatch(ContrastiveLoss):

    def __init__(self, metric='cosine', random_pairs=False, synaptic_scale=0, mean_loss_coef=0):
        """
        :param metric: cosine, l2 or l1 metric to measure the distance between embeddings
        :param random_pairs: select random pairs or use all pairwise combinations
        :param synaptic_scale: synaptic scale constant loss factor to keep neurons activation rate same
        :param mean_loss_coef: coefficient of same-other loss on mean activations only
        """
        super().__init__(metric=metric)
        self.random_pairs = random_pairs
        self.synaptic_scale = synaptic_scale
        self.mean_loss_coef = mean_loss_coef

    def extra_repr(self):
        old_repr = super().extra_repr()
        return f'{old_repr}, random_pairs={self.random_pairs}, synaptic_scale={self.synaptic_scale}, ' \
            f'mean_loss_coef={self.mean_loss_coef}'

    def forward_random(self, outputs, labels, sample_multiplier=2, leave_hardest=0.5):
        n_samples = len(outputs)
        weights = torch.ones(n_samples)
        left_indices = torch.multinomial(weights, sample_multiplier * n_samples, replacement=True).to(device=outputs.device)
        right_indices = torch.multinomial(weights, sample_multiplier * n_samples, replacement=True).to(device=outputs.device)
        dist = self.distance(outputs[left_indices], outputs[right_indices])
        is_same = labels[left_indices] == labels[right_indices]

        dist_same, _unused = dist[is_same].sort(descending=True)
        dist_same = dist_same[: int(len(dist_same) * leave_hardest)]

        dist_other, _unused = dist[~is_same].sort()
        dist_other = dist_other[: int(len(dist_other) * leave_hardest)]

        return dist_same, dist_other

    def forward_pairwise(self, outputs, labels):
        dist_same = []
        dist_other = []
        labels_unique = labels.unique()
        outputs_sorted = {}
        for label in labels_unique:
            outputs_sorted[label.item()] = outputs[labels == label]
        for label_id, label_same in enumerate(labels_unique):
            outputs_same_label = outputs_sorted[label_same.item()]
            n_same = len(outputs_same_label)
            if n_same > 1:
                upper_triangle_idx = np.triu_indices(n=n_same, k=1)
                upper_triangle_idx = torch.as_tensor(upper_triangle_idx, device=outputs_same_label.device)
                same_left, same_right = outputs_same_label[upper_triangle_idx]
                dist_same.append(self.distance(same_left, same_right))

            for label_other in labels_unique[label_id + 1:]:
                outputs_other_label = outputs_sorted[label_other.item()]
                n_other = len(outputs_other_label)
                n_max = max(n_same, n_other)
                idx_same = torch.arange(n_max) % n_same
                idx_other = torch.arange(n_max) % n_other
                dist = self.distance(outputs_other_label[idx_other], outputs_same_label[idx_same])
                dist_other.append(dist)

        dist_same = torch.cat(dist_same)
        dist_other = torch.cat(dist_other)

        return dist_same, dist_other

    def forward(self, outputs, labels):
        outputs, labels = self.filter_nonzero(outputs, labels)
        if self.metric != 'cosine':
            outputs = outputs / outputs.norm(p=self.power, dim=1).mean()
        if self.random_pairs or timer.is_epoch_finished():
            # if an epoch is finished, use random pairs no matter what the mode is
            dist_same, dist_other = self.forward_random(outputs, labels)
        else:
            dist_same, dist_other = self.forward_pairwise(outputs, labels)

        dist_same = dist_same[dist_same > self.eps]
        dist_other = dist_other[dist_other < self.margin]
        loss_same = self.mean_nonempty(dist_same)
        loss_other = self.mean_nonempty(self.margin - dist_other)

        if self.synaptic_scale > 0:
            loss_frequency = self.synaptic_scale * (outputs.mean(dim=0) - outputs.mean()).std()
        else:
            loss_frequency = 0

        if self.mean_loss_coef > 0:
            loss_mean = self.mean_loss_coef * self.forward_mean(outputs, labels)
        else:
            loss_mean = 0

        loss = loss_same + loss_other + loss_frequency + loss_mean

        return loss
