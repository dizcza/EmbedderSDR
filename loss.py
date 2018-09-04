from abc import ABC
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import MAX_L0_DIST


class ContrastiveLoss(nn.Module, ABC):

    def __init__(self, same_only=False, metric='cosine', eps=1e-7):
        """
        :param same_only: use same-only or include same-other classes loss?
        """
        super().__init__()
        self.same_only = same_only
        self.metric = metric
        self.eps = eps
        if self.metric == 'cosine':
            self.margin = 0.5
        else:
            self.margin = 0.7 * MAX_L0_DIST

    def extra_repr(self):
        return f'same_only={self.same_only}, metric={self.metric}, margin={self.margin}'

    def distance(self, input1, input2, is_same: bool):
        if self.metric == 'cosine':
            targets = torch.ones(max(len(input1), len(input2)))
            if not is_same:
                targets *= -1
            return F.cosine_embedding_loss(input1, input2, target=targets, margin=self.margin, reduction='none')
        else:
            diff = input1 - input2
            if self.metric == 'euclidean':
                dist = torch.sum(torch.pow(diff, 2), dim=1)
            elif self.metric == 'l1':
                dist = diff.abs().sum(dim=1)
            else:
                raise NotImplementedError()
            if not is_same:
                dist = torch.max(torch.zeros(len(dist)), self.margin - dist)
            return dist


class ContrastiveLossAnchor(ContrastiveLoss):

    def forward(self, outputs, labels):
        loss = 0
        labels_unique = sorted(labels.unique())
        for label_unique in labels_unique:
            outputs_same_label = outputs[labels == label_unique]
            anchor = outputs_same_label[0].unsqueeze(dim=0)
            if len(outputs_same_label) > 1:
                loss += self.distance(outputs_same_label[1:], anchor, is_same=True).mean()

            if not self.same_only:
                outputs_other_label = outputs[labels != label_unique]
                dist = self.distance(outputs_other_label, anchor, is_same=False)
                loss += dist.mean()

        return loss


class ContrastiveLossBatch(ContrastiveLoss):

    @staticmethod
    def half_majority_mean(distances: torch.FloatTensor):
        if len(distances) == 0:
            return 0
        distances, argsort_ignored = distances.sort(descending=True)
        n_take = math.ceil(len(distances) / 2.)
        return distances[: n_take].mean()

    def forward(self, outputs, labels):
        dist_same = []
        dist_other = []
        outputs_sorted = {}
        labels_unique = labels.unique()
        for label in labels_unique:
            outputs_sorted[label.item()] = outputs[labels == label]
        for label_id, label_same in enumerate(labels_unique):
            outputs_same_label = outputs_sorted[label_same.item()]
            n_same = len(outputs_same_label)
            if n_same > 1:
                dist = self.distance(outputs_same_label[1:], outputs_same_label[:-1], is_same=True)
                dist_same.append(dist)

            if not self.same_only:
                for label_other in labels_unique[label_id+1:]:
                    outputs_other_label = outputs_sorted[label_other.item()]
                    n_other = len(outputs_other_label)
                    n_max = max(n_same, n_other)
                    idx_same = torch.arange(n_max) % n_same
                    idx_other = torch.arange(n_max) % n_other
                    dist = self.distance(outputs_other_label[idx_other], outputs_same_label[idx_same], is_same=False)
                    dist = dist[dist > self.eps]
                    dist_other.append(dist)

        # loss_same = self.half_majority_mean(torch.cat(dist_same))
        loss_same = torch.cat(dist_same).mean()
        dist_other = tuple(filter(len, dist_other))
        if len(dist_other) > 0:
            loss_other = torch.cat(dist_other).mean()
            # loss_other = self.half_majority_mean(torch.cat(dist_other))
        else:
            loss_other = 0
        loss = loss_same + loss_other

        return loss
