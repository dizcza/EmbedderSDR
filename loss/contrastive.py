import math
from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monitor.batch_timer import timer
from utils.layers import SerializableModule
from utils.algebra import compute_distance


class PairLoss(nn.Module, ABC):

    def __init__(self, metric='cosine', margin: float = None, pairs_multiplier: int = 1, leave_hardest: float = 1.0):
        """
        :param metric: cosine, l2 or l1 metric to measure the distance between embeddings
        :param margin: same-other margin
        :param pairs_multiplier: how many pairs create from a single sample?
        :param leave_hardest: hard negative & positive mining
        """
        assert 0 < leave_hardest <= 1, "Value should be in (0, 1]"
        super().__init__()
        self.metric = metric
        if margin is None:
            if self.metric == 'cosine':
                margin = 0.5
            else:
                margin = 1.0
        self.margin = margin
        self.leave_hardest = leave_hardest
        self.pairs_multiplier = pairs_multiplier

    @property
    def power(self):
        if self.metric == 'l1':
            return 1
        elif self.metric == 'l2':
            return 2
        else:
            raise NotImplementedError

    def extra_repr(self):
        return f'metric={self.metric}, margin={self.margin}, ' \
            f'pairs_multiplier={self.pairs_multiplier}, leave_hardest={self.leave_hardest}'

    def filter_nonzero(self, outputs, labels, normalize: bool):
        nonzero = (outputs != 0).any(dim=1)
        outputs = outputs[nonzero]
        labels = labels[nonzero]
        if normalize and self.metric != 'cosine':
            # sparsity changes with epoch but margin stays the same
            outputs = outputs / outputs.norm(p=self.power, dim=1).mean()
        return outputs, labels

    @staticmethod
    def mean_nonempty(distances):
        return 0 if len(distances) == 0 else distances.mean()

    def distance(self, input1, input2):
        return compute_distance(input1=input1, input2=input2, metric=self.metric, dim=1)

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

    def pairs_to_sample(self, labels):
        """
        Probability of two random samples having same class is 1/n_classes.
        On average, each single sample in a batch produces 1/n_classes pairs
          or 1/n_classes * (1 - 1/n_classes) triplets.
        :param labels: batch of labels
        :return: how many random permutations to sample to get the desired number of pairs or triplets
        """
        batch_size = len(labels)
        n_unique = len(labels.unique(sorted=False))
        random_pairs_shape = (self.pairs_multiplier * n_unique * batch_size,)
        return random_pairs_shape

    def take_hardest(self, distances):
        if self.leave_hardest < 1.0:
            distances, _unused = distances.sort(descending=True)
            distances = distances[: int(len(distances) * self.leave_hardest)]
        return distances


class LossFixedPattern(PairLoss, SerializableModule):
    state_attr = ['patterns']

    def __init__(self, sparsity: float, metric='cosine'):
        super().__init__(metric=metric)
        self.patterns = {}
        self.sparsity = sparsity

    def forward(self, outputs, labels):
        outputs, labels = self.filter_nonzero(outputs, labels, normalize=False)  # sparsity is fixed
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


class ContrastiveLossRandom(PairLoss):

    def __init__(self, metric='cosine', margin: float = None, pairs_multiplier: int = 1, leave_hardest: float = 1.0,
                 margin_same: float = 0.1, synaptic_scale: float = 0, mean_loss_coef: float = 0):
        """
        :param metric: cosine, l2 or l1 metric to measure the distance between embeddings
        :param margin: same-other margin
        :param pairs_multiplier: how many pairs create from a single sample?
        :param leave_hardest: hard negative & positive mining
        :param margin_same: same-same margin below which you pay zero loss
        :param synaptic_scale: synaptic scale constant loss factor to keep neurons activation rate same
        :param mean_loss_coef: coefficient of same-other loss on mean activations only
        """
        super().__init__(metric=metric, margin=margin, pairs_multiplier=pairs_multiplier, leave_hardest=leave_hardest)
        self.margin_same = margin_same
        self.synaptic_scale = synaptic_scale
        self.mean_loss_coef = mean_loss_coef

    def extra_repr(self):
        old_repr = super().extra_repr()
        return f'{old_repr}, margin_same={self.margin_same}, synaptic_scale={self.synaptic_scale}, ' \
            f'mean_loss_coef={self.mean_loss_coef}'

    def forward_contrastive(self, outputs, labels):
        return self.forward_random(outputs, labels)

    def forward_random(self, outputs, labels):
        n_samples = len(outputs)
        pairs_to_sample = self.pairs_to_sample(labels)
        left_indices = torch.randint(low=0, high=n_samples, size=pairs_to_sample, device=outputs.device)
        right_indices = torch.randint(low=0, high=n_samples, size=pairs_to_sample, device=outputs.device)
        dist = self.distance(outputs[left_indices], outputs[right_indices])
        is_same = labels[left_indices] == labels[right_indices]

        dist_same = dist[is_same]
        dist_other = dist[~is_same]

        return dist_same, dist_other

    def forward(self, outputs, labels):
        outputs, labels = self.filter_nonzero(outputs, labels, normalize=True)
        if timer.is_epoch_finished():
            # if an epoch is finished, use random pairs no matter what the mode is
            dist_same, dist_other = self.forward_random(outputs, labels)
        else:
            dist_same, dist_other = self.forward_contrastive(outputs, labels)

        dist_same = self.take_hardest(dist_same)
        loss_same = torch.relu(dist_same - self.margin_same).mean()

        loss_other = self.margin - dist_other
        loss_other = self.take_hardest(loss_other)
        loss_other = torch.relu(loss_other).mean()

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


class ContrastiveLossPairwise(ContrastiveLossRandom):

    def forward_contrastive(self, outputs, labels):
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
