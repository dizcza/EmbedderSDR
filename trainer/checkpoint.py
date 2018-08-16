import shutil
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

from constants import CHECKPOINTS_DIR


class Checkpoint:
    def __init__(self, model: nn.Module, patience: int = None):
        """
        :param model: trained model
        :param patience: # of steps to allow loss to be larger than the best loss so far
        """
        self.patience = patience
        self.best_loss = float('inf')
        self.num_bad_epochs = 0
        checkpoint_dir = CHECKPOINTS_DIR.joinpath(time.strftime('%Y-%b-%d'))
        if checkpoint_dir.exists():
            shutil.rmtree(path=checkpoint_dir)
        checkpoint_dir.mkdir(parents=True)
        self.best_model_path = checkpoint_dir.joinpath(time.strftime('%H-%M-%S_init.pt'))
        self._save(model)

    def _save(self, model: nn.Module):
        torch.save(model.state_dict(), self.best_model_path)

    def is_active(self) -> bool:
        return self.patience is not None

    def step(self, model: nn.Module, loss: torch.Tensor):
        if self.is_active() and loss < self.best_loss:
            self.best_loss = loss
            self.num_bad_epochs = 0
            self.best_model_path = self.best_model_path.with_name(f"{time.strftime('%H-%M-%S')}_loss={loss.data[0]}.pt")
            self._save(model)
        else:
            self.num_bad_epochs += 1

    def need_reset(self) -> bool:
        return self.is_active() and self.num_bad_epochs > self.patience

    def reset(self, model: nn.Module):
        assert self.is_active()
        self.num_bad_epochs = 0
        model_state = torch.load(self.best_model_path)
        model.load_state_dict(model_state)
