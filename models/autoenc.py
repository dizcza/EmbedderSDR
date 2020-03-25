import torch.nn as nn

from models.kwta import KWinnersTakeAllSoft


class AutoEncoderLinear(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int, sparsity=0.15):
        super().__init__()
        # self.encoder = nn.Sequential(nn.Linear(input_dim, encoding_dim), nn.ReLU(inplace=True))
        self.encoder = nn.Linear(input_dim, encoding_dim, bias=False)
        self.decoder = nn.Linear(encoding_dim, input_dim)
        self.kwta = KWinnersTakeAllSoft(hardness=1, sparsity=sparsity)

    def forward(self, x):
        input_shape = x.shape
        x = x.flatten(start_dim=1)
        encoded = self.encoder(x)
        encoded = self.kwta(encoded)
        decoded = self.decoder(encoded)
        decoded = decoded.view(*input_shape)
        return decoded


class AutoEncoderConv(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int, sparsity=0.15):
        super().__init__()
        conv_channels = 3
        linear_in_features = conv_channels * 8 * 8
        self.conv = nn.Conv2d(in_channels=1, out_channels=conv_channels, kernel_size=3, bias=False)
        self.bn = nn.BatchNorm2d(num_features=conv_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3)
        self.encoder = nn.Linear(linear_in_features, encoding_dim, bias=False)
        self.decoder = nn.Linear(encoding_dim, input_dim)
        self.kwta = KWinnersTakeAllSoft(hardness=1, sparsity=sparsity)

    def forward(self, x):
        input_shape = x.shape
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.flatten(start_dim=1)
        x = self.encoder(x)
        encoded = self.kwta(x)
        decoded = self.decoder(encoded)
        decoded = decoded.view(*input_shape)
        return decoded
