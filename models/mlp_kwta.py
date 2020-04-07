import torch.nn as nn

from .kwta import KWinnersTakeAll


class MLP_kWTA(nn.Module):
    """
    Creates sequential fully-connected layers FC_1->FC_2->...->FC_N.

    Parameters
    ----------
    fc_sizes : int
        Fully connected sequential layer sizes.
    """

    def __init__(self, fc1, fc2, kwta: KWinnersTakeAll):
        super().__init__()
        self.linear = nn.Linear(fc1, fc2, bias=False)
        self.kwta = kwta

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.linear(x)
        x = self.kwta(x)
        return x


class MLP_kWTA_Autoenc(MLP_kWTA):

    def forward(self, x):
        input_shape = x.shape
        encoded = super().forward(x)
        reconstructed = encoded.matmul(self.weight).view(*input_shape)
        return encoded, reconstructed
