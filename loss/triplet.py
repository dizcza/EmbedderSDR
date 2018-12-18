import torch
import torch.nn.functional as F

from loss.contrastive import PairLoss


class TripletLoss(PairLoss):

    def forward(self, outputs, labels):
        outputs, labels = self.filter_nonzero(outputs, labels, normalize=True)
        n_samples = len(outputs)
        n_unique = len(labels.unique(sorted=False))  # probability of two random samples having same class is 1/n_unique
        random_sample_shape = (n_unique * n_samples,)
        anchor = torch.randint(low=0, high=n_samples, size=random_sample_shape, device=outputs.device)
        same = torch.randint(low=0, high=n_samples, size=random_sample_shape, device=outputs.device)
        other = torch.randint(low=0, high=n_samples, size=random_sample_shape, device=outputs.device)

        triplets = (labels[anchor] == labels[same]) & (labels[anchor] != labels[other])
        anchor = anchor[triplets]
        same = same[triplets]
        other = other[triplets]

        if self.metric == 'cosine':
            dist_same = self.distance(outputs[anchor], outputs[same])
            dist_other = self.distance(outputs[anchor], outputs[other])
            loss = dist_same - dist_other + self.margin
            loss = torch.relu(loss)
        else:
            loss = F.triplet_margin_loss(outputs[anchor], outputs[same], outputs[other], margin=self.margin,
                                         p=self.power, reduction='none')
        loss = self.take_hardest(loss).mean()

        return loss
