from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision

from models.kwta import KWinnersTakeAll

SOFTMAX_FEATURES = 256
KWTA_FEATURES = 512


def _classifier(in_features: int, kwta: Optional[nn.Module]):
    if kwta:
        return nn.Sequential(nn.Linear(in_features=in_features, out_features=KWTA_FEATURES, bias=False), kwta)
    else:
        return nn.Linear(in_features=in_features, out_features=SOFTMAX_FEATURES)


def make_no_grad(model):
    for param in model.parameters():
        param.requires_grad_(False)


def set_softmax_out_features(out_features):
    global SOFTMAX_FEATURES
    SOFTMAX_FEATURES = out_features


def resnet18(pretrained=True, kwta=None):
    model = torchvision.models.resnet18(pretrained=pretrained)
    make_no_grad(model)
    model.fc = _classifier(in_features=512, kwta=kwta)
    return model


def squeezenet1_1(pretrained=True, kwta=None):
    model = torchvision.models.squeezenet1_1(pretrained=pretrained)
    make_no_grad(model)
    model.num_classes = SOFTMAX_FEATURES
    final_conv = nn.Conv2d(512, model.num_classes, kernel_size=1)
    nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
    classifier = [nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AvgPool2d(13, stride=1)]
    if kwta:
        classifier.append(kwta)
    model.classifier = nn.Sequential(*classifier)
    return model


def densenet121(pretrained=True, kwta=None):
    model = torchvision.models.densenet121(pretrained=pretrained)
    make_no_grad(model)
    model.classifier = _classifier(in_features=1024, kwta=kwta)
    return model


def vgg16(pretrained=True, kwta: KWinnersTakeAll = None, checkpoint: Path = None):
    """
    :param pretrained: use pretrained on ImageNet?
    :param kwta: the last layer is kwta or softmax?
    :param checkpoint: checkpoint path to TrainerGrad CrossEntropy, where the last layer is softmax
    :return: model for Caltech dataset
    """
    if checkpoint is not None:
        model = vgg16(kwta=None, checkpoint=None)
        checkpoint_state = torch.load(checkpoint)
        model.load_state_dict(checkpoint_state['model_state'])
    else:
        model = torchvision.models.vgg16(pretrained=pretrained)
    make_no_grad(model)
    classifier = [*model.classifier]
    last_layer = classifier.pop()  # 4096
    classifier.append(_classifier(in_features=last_layer.in_features, kwta=kwta))
    classifier = nn.Sequential(*classifier)
    model.classifier = classifier
    return model
