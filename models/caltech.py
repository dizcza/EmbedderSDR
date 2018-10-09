from typing import Optional

import torch.nn as nn
import torchvision


def _classifier(in_features: int, kwta: Optional[nn.Module]):
    if kwta:
        return nn.Sequential(nn.Linear(in_features=in_features, out_features=256, bias=False), kwta)
    else:
        return nn.Linear(in_features=in_features, out_features=256)


def resnet18(pretrained=True, kwta=None):
    model = torchvision.models.resnet18(pretrained=pretrained)
    model.fc = _classifier(in_features=512, kwta=kwta)
    return model


def squeezenet1_1(pretrained=True, kwta=None):
    model = torchvision.models.squeezenet1_1(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad_(False)
    model.num_classes = 256
    final_conv = nn.Conv2d(512, model.num_classes, kernel_size=1)
    classifier = [nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AvgPool2d(13, stride=1)]
    if kwta:
        classifier.append(kwta)
    model.classifier = nn.Sequential(*classifier)
    nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
    return model


def densenet121(pretrained=True, kwta=None):
    model = torchvision.models.densenet121(pretrained=pretrained)
    model.classifier = _classifier(in_features=1024, kwta=kwta)
    return model


def vgg19(pretrained=True, kwta=None):
    model = torchvision.models.vgg19(pretrained=pretrained)
    classifier = [*model.classifier]
    last_layer = classifier.pop()  # 4096
    model.classifier = _classifier(in_features=last_layer.in_features, kwta=kwta)
    return model
