import re
from pathlib import Path

import torch
import torch.nn as nn

from utils.constants import CHECKPOINTS_DIR


class Checkpoint:
    def __init__(self, checkpoint_dir: Path):
        self.env_name = None
        self.best_loss = float('inf')
        self.epoch = 0
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_path = self.find_best_model_path()

    def find_best_model_path(self):
        best_loss = float('inf')
        best_model_path = None
        for model_path in self.checkpoint_dir.iterdir():
            match = re.search("loss=.*", model_path.stem)
            if match:
                loss = float(match.group().lstrip("loss="))
                if loss < best_loss:
                    best_loss = loss
                    best_model_path = model_path
        return best_model_path

    def _save(self, model: nn.Module):
        torch.save({
            "model_state": model.state_dict(),
            "loss": self.best_loss,
            "epoch": self.epoch,
            "env_name": self.env_name,
        }, self.best_model_path)

    def epoch_finished(self, model: nn.Module, loss: torch.Tensor, epoch: int):
        self.epoch = epoch
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_model_path = self.checkpoint_dir / f"epoch={self.epoch:03d}_loss={loss.item():.5f}.pt"
            self._save(model)

    def restore(self, model: nn.Module):
        if self.best_model_path is not None:
            checkpoint = torch.load(self.best_model_path)
            try:
                model.load_state_dict(checkpoint['model_state'])
            except RuntimeError as error:
                print("Error is occurred while restoring model: ", error)
                return False
            self.env_name = checkpoint['env_name']
            self.best_loss = checkpoint['loss']
            self.epoch = checkpoint['epoch']
            print(f"Restored model state from {self.best_model_path}.")
            return True
        else:
            print(f"Couldn't find saved model state. Nothing to restore.")
            return False
