import copy

import torch.nn as nn


class Identity(nn.Module):
    def forward(self, x):
        return x


def replace_relu(model: nn.Module, new_relu, drop_layers=()):
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
