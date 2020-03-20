import torch.nn as nn

from .kwta import KWinnersTakeAll


class MLPKwta(nn.Module):
    """
    Creates sequential fully-connected layers FC_1->FC_2->...->FC_N.

    Parameters
    ----------
    fc_sizes : int
        Fully connected sequential layer sizes.
    """

    def __init__(self, fc1, fc2, kwta_layer: KWinnersTakeAll):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(fc1, fc2),
            kwta_layer,
        )

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x
