import torch.nn as nn


class MLP(nn.Module):

    def __init__(self, *fc_sizes: int):
        """
        :param fc_sizes: fully-connected layer sizes
        """
        super().__init__()
        classifier = []
        for in_features, out_features in zip(fc_sizes[:-1], fc_sizes[1:]):
            linear = nn.Linear(in_features=in_features, out_features=out_features)
            classifier.append(linear)
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x
