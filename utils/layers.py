import copy

import torch.nn as nn

from model.kwta import KWinnersTakeAll


class Identity(nn.Module):
    def forward(self, x):
        return x


def find_layers(model: nn.Module, layer_class):
    for name, layer in find_named_layers(model, layer_class=layer_class):
        yield layer


def find_named_layers(model: nn.Module, layer_class, name_prefix=''):
    for name, layer in model.named_children():
        yield from find_named_layers(layer, layer_class, name_prefix=f"{name_prefix}.{name}")
    if isinstance(model, layer_class):
        yield name_prefix.lstrip('.'), model


def replace_relu(model: nn.Module, new_relu: KWinnersTakeAll, drop_layers=()):
    """
    :param model: network model
    :param new_relu: new relu activation function
    :param drop_layers: drop these layers; you can try dropping batch norm
    :return: model with relu replaced by kWTA activation function
    """
    if isinstance(model, new_relu.__class__):
        return model
    for name, child in list(model.named_children()):
        if isinstance(child, drop_layers):
            setattr(model, name, Identity())
            continue
        child_new = replace_relu(model=child, new_relu=new_relu, drop_layers=drop_layers)
        if child_new is not child:
            setattr(model, name, child_new)
    if isinstance(model, (nn.ReLU, nn.RReLU, nn.ReLU6, nn.LeakyReLU, nn.PReLU)):
        model = copy.deepcopy(new_relu)
    return model
