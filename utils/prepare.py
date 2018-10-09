import torch.nn as nn

from models.kwta import KWinnersTakeAllSoft
from utils.layers import find_layers


class ModelMode:
    def __init__(self, mode: bool, requires_grad: dict):
        self.mode = mode
        self.requires_grad = requires_grad

    def restore(self, model: nn.Module):
        model.train(self.mode)
        for name, param in model.named_parameters():
            param.requires_grad_(self.requires_grad[name])


def prepare_eval(model: nn.Module):
    mode_saved = model.training
    requires_grad_saved = {}
    model.eval()
    for kwta in find_layers(model, KWinnersTakeAllSoft):
        kwta.train()  # use soft kwta to pass gradient
    for name, param in model.named_parameters():
        requires_grad_saved[name] = param.requires_grad
        param.requires_grad_(False)
    return ModelMode(mode=mode_saved, requires_grad=requires_grad_saved)
