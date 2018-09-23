from abc import ABC

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monitor.batch_timer import timer


class ContrastiveLoss(nn.Module, ABC):

    def __init__(self, metric='cosine', eps=1e-7):
        """
        :param metric: cosine, l2 or l1 metric to measure the distance between embeddings
        :param eps: threshold to skip negligible same-other loss
        """
        super().__init__()
        self.metric = metric
        self.eps = eps
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

    def distance(self, input1, input2, is_same: bool):
        if self.metric == 'cosine':
            targets = torch.ones(max(len(input1), len(input2)), device=input1.device)
            if not is_same:
                targets *= -1
            return F.cosine_embedding_loss(input1, input2, target=targets, margin=self.margin, reduction='none')
        else:
            dist = (input1 - input2).norm(p=self.power, dim=1)
            if not is_same:
                zeros = torch.zeros_like(dist, device=dist.device)
                dist = torch.max(zeros, self.margin - dist)
            return dist


class ContrastiveLossBatch(ContrastiveLoss):

    def __init__(self, metric='cosine', eps=1e-7, random_pairs=False):
        """
        :param metric: cosine, l2 or l1 metric to measure the distance between embeddings
        :param eps: threshold to skip negligible same-other loss
        :param random_pairs: select random pairs or use all pairwise combinations
        """
        super().__init__(metric=metric, eps=eps)
        self.random_pairs = random_pairs

    def extra_repr(self):
        old_repr = super().extra_repr()
        return f'{old_repr}, random_pairs={self.random_pairs}'

    def forward_random(self, outputs, labels):
        dist_same = []
        dist_other = []
        labels_unique = labels.unique()
        for label_id, label_same in enumerate(labels_unique):
            mask_same = labels == label_same
            outputs_same_label = outputs[mask_same]
            n_same = len(outputs_same_label)
            if n_same > 1:
                dist = self.distance(outputs_same_label[1:], outputs_same_label[:-1], is_same=True)
                dist_same.append(dist)

            other_idx = np.arange(len(labels))[~mask_same.cpu()]
            other_idx = np.random.choice(other_idx, size=n_same, replace=True)
            other_idx = torch.as_tensor(other_idx, device=outputs.device)
            outputs_other_label = outputs[other_idx]
            dist = self.distance(outputs_other_label, outputs_same_label, is_same=False)
            dist_other.append(dist)
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
                dist_same.append(self.distance(same_left, same_right, is_same=True))

            for label_other in labels_unique[label_id + 1:]:
                outputs_other_label = outputs_sorted[label_other.item()]
                n_other = len(outputs_other_label)
                n_max = max(n_same, n_other)
                idx_same = torch.arange(n_max) % n_same
                idx_other = torch.arange(n_max) % n_other
                dist = self.distance(outputs_other_label[idx_other], outputs_same_label[idx_same], is_same=False)
                dist_other.append(dist)
        return dist_same, dist_other

    def forward(self, outputs, labels):
        nonzero = (outputs != 0).any(dim=1)
        outputs = outputs[nonzero]
        labels = labels[nonzero]
        if self.metric != 'cosine':
            outputs = outputs / outputs.norm(p=self.power, dim=1).mean()
        if self.random_pairs or timer.is_epoch_finished():
            # if an epoch is finished, use random pairs no matter what the mode is
            dist_same, dist_other = self.forward_random(outputs, labels)
        else:
            dist_same, dist_other = self.forward_pairwise(outputs, labels)

        loss_same = torch.cat(dist_same).mean()
        dist_other = torch.cat(dist_other)
        dist_other = dist_other[dist_other > self.eps]
        if len(dist_other) > 0:
            loss_other = dist_other.mean()
        else:
            loss_other = 0
        loss = loss_same + loss_other

        return loss
