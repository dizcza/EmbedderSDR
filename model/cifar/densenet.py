import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, last_layer: nn.Module, n_blocks_layers: list, growth_rate=12, reduction=0.5):
        super().__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        feature_layers = []
        for n_blocks in n_blocks_layers[:3]:
            dense = self.make_dense_layers(num_planes, n_blocks)
            num_planes += n_blocks * growth_rate
            out_planes = math.floor(num_planes * reduction)
            trans = Transition(num_planes, out_planes)
            feature_layers.extend([dense, trans])
            num_planes = out_planes
        dense4 = self.make_dense_layers(num_planes, n_blocks_layers[3])
        num_planes += n_blocks_layers[3] * growth_rate
        feature_layers.append(dense4)
        self.features = nn.Sequential(*feature_layers)

        self.bn = nn.BatchNorm2d(num_planes)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool2d = nn.AvgPool2d(kernel_size=4)
        self.last_layer = last_layer

    def make_dense_layers(self, in_planes, n_blocks):
        layers = []
        for i in range(n_blocks):
            layers.append(Bottleneck(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.features(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        if self.last_layer is not None:
            out = self.last_layer(out)
        return out


class DenseNet121(DenseNet):
    def __init__(self, last_layer):
        super().__init__(last_layer=last_layer, n_blocks_layers=[6, 12, 24, 16], growth_rate=32)


class DenseNet169(DenseNet):
    def __init__(self, last_layer):
        super().__init__(last_layer=last_layer, n_blocks_layers=[6, 12, 48, 32], growth_rate=32)


class DenseNet201(DenseNet):
    def __init__(self, last_layer):
        super().__init__(last_layer=last_layer, n_blocks_layers=[6, 12, 48, 32], growth_rate=32)


class DenseNet161(DenseNet):
    def __init__(self, last_layer):
        super().__init__(last_layer=last_layer, n_blocks_layers=[6, 12, 36, 24], growth_rate=48)
