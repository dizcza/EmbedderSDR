import torch.nn as nn

from constants import SPARSITY, EMBEDDING_SIZE
from model.kwta import KWinnersTakeAll


class EmbedderSDR(nn.Module):

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc_emb(x)
        x = self.relu2(x)
        if self.kwta is not None:
            x = self.kwta(x)
        return x

    def __init__(self, kwta_layer: KWinnersTakeAll, dataset_name="MNIST", conv_channels=3):
        super().__init__()
        if "MNIST" in dataset_name:
            conv_in_channels = 1
            linear_in_features = conv_channels * 8 * 8
        elif "CIFAR10" in dataset_name:
            conv_in_channels = 3
            linear_in_features = conv_channels * 10 * 10
        else:
            raise NotImplementedError()
        self.conv = nn.Conv2d(in_channels=conv_in_channels, out_channels=conv_channels, kernel_size=3, bias=False)
        self.bn = nn.BatchNorm2d(num_features=conv_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3)
        self.fc_emb = nn.Linear(in_features=linear_in_features, out_features=EMBEDDING_SIZE, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        self.kwta = kwta_layer
