import copy
from abc import ABC

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


class SerializableModule(nn.Module, ABC):
    state_attr = []

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        destination = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        for attribute in self.state_attr:
            destination[prefix + attribute] = getattr(self, attribute)
        return destination

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        state_dict_keys = list(state_dict.keys())
        for attribute in self.state_attr:
            key = prefix + attribute
            if key in state_dict_keys:
                setattr(self, attribute, state_dict.pop(key))
            elif strict:
                missing_keys.append(key)
        super()._load_from_state_dict(state_dict=state_dict, prefix=prefix, local_metadata=local_metadata,
                                      strict=strict, missing_keys=missing_keys, unexpected_keys=unexpected_keys,
                                      error_msgs=error_msgs)
