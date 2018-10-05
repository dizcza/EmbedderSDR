import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

CIFAR10_PRETRAINED_URL = 'http://ml.cs.tsinghua.edu.cn/~chenxi/pytorch-models/cifar10-d875770b.pth'


def make_feature_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            layers.append(conv2d)
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels, affine=False))
            layers.append(nn.ReLU())
            in_channels = out_channels
    return nn.Sequential(*layers)


class CIFAR10(nn.Module):
    def __init__(self, n_channel=128, num_classes=10, pretrained=False):
        super().__init__()
        cfg = [n_channel, n_channel, 'M', 2 * n_channel, 2 * n_channel, 'M', 4 * n_channel, 4 * n_channel, 'M',
               (8 * n_channel, 0), 'M']
        self.features = make_feature_layers(cfg, batch_norm=True)
        self.classifier = nn.Sequential(
            nn.Linear(8 * n_channel, num_classes)
        )
        if pretrained:
            map_location = None
            if not torch.cuda.is_available():
                map_location = 'cpu'
            state_dict = model_zoo.load_url(CIFAR10_PRETRAINED_URL, map_location=map_location)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
