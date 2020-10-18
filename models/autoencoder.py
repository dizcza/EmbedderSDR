from typing import Iterable, Union

import torch.nn as nn

from models import KWinnersTakeAll
from mighty.models import AutoencoderOutput, Flatten


class AutoencoderLinearKWTA(nn.Module):
    def __init__(self,
                 input_dim: Union[int, Iterable[int]],
                 encoding_dim: int,
                 kwta: KWinnersTakeAll = None):
        super().__init__()
        self.encoding_dim = encoding_dim
        if isinstance(input_dim, int):
            input_dim = [input_dim]
        else:
            input_dim = list(input_dim)
        encoder = []
        for in_features, out_features in zip(input_dim[:-1], input_dim[1:]):
            encoder.append(nn.Linear(in_features, out_features))
            encoder.append(nn.ReLU(inplace=True))
        encoder.append(nn.Linear(input_dim[-1],
                                 encoding_dim, bias=False))
        if kwta:
            encoder.append(kwta)
        else:
            encoder.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(Flatten(), *encoder)

        self.decoder = nn.Linear(encoding_dim, input_dim[0])

    def forward(self, x):
        input_shape = x.shape
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(*input_shape)
        return AutoencoderOutput(encoded, decoded)


class AutoencoderLinearKWTATanh(AutoencoderLinearKWTA):
    def forward(self, x):
        encoded, decoded = super().forward(x)
        decoded = decoded.tanh()
        return AutoencoderOutput(encoded, decoded)


class AutoencoderConvKWTA(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int,
                 kwta: KWinnersTakeAll = None):
        super().__init__()
        self.encoding_dim = encoding_dim
        conv_channels = 3
        linear_in_features = conv_channels * 8 * 8
        self.conv = nn.Conv2d(in_channels=1, out_channels=conv_channels,
                              kernel_size=3, bias=False)
        self.bn = nn.BatchNorm2d(num_features=conv_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3)

        encoder = [nn.Linear(linear_in_features, encoding_dim, bias=False)]
        if kwta:
            encoder.append(kwta)
        else:
            encoder.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*encoder)

        self.decoder = nn.Linear(encoding_dim, input_dim)

    def forward(self, x):
        input_shape = x.shape
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.flatten(start_dim=1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(*input_shape)
        return AutoencoderOutput(encoded, decoded)
