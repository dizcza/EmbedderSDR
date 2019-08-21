import pickle
import shutil
from pathlib import Path

import torch.nn as nn

from utils.constants import DUMPS_DIR


class LayersOrderHook:
    def __init__(self, model: nn.Module):
        self.hooks = []
        self.layers_ordered = []
        self.register_hooks(model)

    def register_hooks(self, model: nn.Module):
        children = tuple(model.children())
        if any(children):
            for layer in children:
                self.register_hooks(layer)
        else:
            handle = model.register_forward_pre_hook(self.append_layer)
            self.hooks.append(handle)

    def append_layer(self, layer, tensor_input):
        self.layers_ordered.append(layer)

    def get_layers_ordered(self):
        for handle in self.hooks:
            handle.remove()
        return tuple(self.layers_ordered)


class DumpActivationsHook:
    def __init__(self, model: nn.Module,
                 inspect_layers=(nn.Linear, nn.Conv2d),
                 dumps_dir=DUMPS_DIR):
        self.hooks = []
        self.layer_to_name = {}
        self.inspect_layers = inspect_layers
        self.dumps_dir = Path(dumps_dir) / model._get_name()
        shutil.rmtree(self.dumps_dir, ignore_errors=True)
        self.dumps_dir.mkdir(parents=True)
        self.register_hooks(model)
        print(f"Dumping activations from {self.layer_to_name.values()} layers "
              f"to {self.dumps_dir}.")

    def register_hooks(self, model: nn.Module, prefix=''):
        children = tuple(model.named_children())
        if any(children):
            for name, layer in children:
                self.register_hooks(layer, prefix=f"{prefix}.{name}")
        elif isinstance(model, self.inspect_layers):
            self.layer_to_name[model] = prefix.lstrip('.')
            handle = model.register_forward_hook(self.dump_activations)
            self.hooks.append(handle)

    def dump_activations(self, layer, tensor_input, tensor_output):
        layer_name = self.layer_to_name[layer]
        layer_path = self.dumps_dir / layer_name
        activations_input_path = f"{layer_path}_inp.pkl"
        activations_output_path = f"{layer_path}_out.pkl"
        if isinstance(tensor_input, tuple):
            assert len(tensor_input) == 1, "Expected only 1 input tensor"
            tensor_input = tensor_input[0]
        with open(activations_input_path, 'ab') as f:
            pickle.dump(tensor_input.detach().cpu(), f)
        with open(activations_output_path, 'ab') as f:
            pickle.dump(tensor_output.detach().cpu(), f)

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
