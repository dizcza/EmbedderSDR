from typing import Iterable, Union

import torch.nn as nn


class AutoEncoderLinear(nn.Module):
    def __init__(self,
                 input_dim: Union[int, Iterable[int]],
                 encoding_dim: int,
                 kwta):
        super().__init__()
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
        encoder.append(kwta)
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Linear(encoding_dim, input_dim[0])

    def forward(self, x):
        input_shape = x.shape
        x = x.flatten(start_dim=1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(*input_shape)
        return encoded, decoded


class AutoEncoderConv(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int, kwta):
        super().__init__()
        conv_channels = 3
        linear_in_features = conv_channels * 8 * 8
        self.conv = nn.Conv2d(in_channels=1, out_channels=conv_channels,
                              kernel_size=3, bias=False)
        self.bn = nn.BatchNorm2d(num_features=conv_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3)
        self.encoder = nn.Sequential(
            nn.Linear(linear_in_features, encoding_dim, bias=False),
            kwta
        )
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
        return encoded, decoded
