from mighty.models import MLP
from .kwta import KWinnersTakeAll


class MLP_kWTA(MLP):
    """
    Creates sequential fully-connected layers FC_1->FC_2->...->FC_N.

    Parameters
    ----------
    fc_sizes : int
        Fully connected sequential layer sizes.
    """

    def __init__(self, *fc_sizes: int, kwta: KWinnersTakeAll):
        super().__init__(*fc_sizes)
        self.kwta = kwta

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.mlp(x)
        x = self.kwta(x)
        return x
