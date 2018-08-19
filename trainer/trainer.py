import copy
import time
import warnings
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from constants import MODELS_DIR, ROOT_DIR
from monitor.accuracy import calc_accuracy, get_outputs
from monitor.batch_timer import timer
from monitor.monitor import Monitor
from trainer.checkpoint import Checkpoint
from utils import get_data_loader, load_model_state


class Trainer(ABC):

    watch_modules = (nn.Linear, nn.Conv2d)

    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str, patience=None,
                 project_name=ROOT_DIR.name):
        self.model = model
        self.criterion = criterion
        self.dataset_name = dataset_name
        self.train_loader = get_data_loader(dataset_name, train=True)
        timer.init(batches_in_epoch=len(self.train_loader))
        env_name = f"{time.strftime('%Y.%m.%d')} {project_name}: {self.dataset_name} {self.__class__.__name__}"
        self.monitor = Monitor(test_loader=get_data_loader(self.dataset_name, train=False), env_name=env_name)
        self._monitor_parameters(self.model)
        self.checkpoint = Checkpoint(model=self.model, patience=patience)

    def log_trainer(self):
        self.monitor.log(f"Criterion: {self.criterion}")

    def save_model(self, accuracy: float = None):
        model_path = MODELS_DIR.joinpath(self.dataset_name, self.model.__class__.__name__).with_suffix('.pt')
        model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), model_path)
        msg = f"Saved to {model_path}"
        if accuracy is not None:
            msg += f" (train accuracy: {accuracy:.4f})"
        print(msg)

    def reset_checkpoint(self):
        self.checkpoint.reset(model=self.model)

    def load_best_accuracy(self) -> float:
        best_accuracy = 0.
        try:
            model_state = load_model_state(self.dataset_name, self.model.__class__.__name__)
            loaded_model = copy.deepcopy(self.model)
            loaded_model.load_state_dict(model_state)
            loaded_model.eval()
            best_accuracy = calc_accuracy(loaded_model, self.train_loader)
            del loaded_model
        except Exception as e:
            print(f"Couldn't estimate the best accuracy for {self.model.__class__.__name__}. Reset to 0.")
        return best_accuracy

    def _monitor_parameters(self, layer: nn.Module, prefix=''):
        for name, child in layer.named_children():
            self._monitor_parameters(child, prefix=f'{prefix}.{name}')
        if isinstance(layer, self.watch_modules):
            self.monitor.register_layer(layer, prefix=prefix.lstrip('.'))

    @abstractmethod
    def train_batch(self, images, labels):
        raise NotImplementedError()

    def _epoch_finished(self, epoch, outputs, labels) -> torch.Tensor:
        loss = self.criterion(outputs, labels).item()
        self.monitor.update_loss(loss, mode='full train')
        self.checkpoint.step(model=self.model, loss=loss)
        return loss

    def train(self, n_epoch=10, save=True, epoch_update_step=1, with_mutual_info=False, watch_parameters=False):
        """
        :param n_epoch: number of training epochs
        :param save: save the trained model?
        :param epoch_update_step: epoch step to run full evaluation
        :param with_mutual_info: plot the mutual information of layer activations?
        :param watch_parameters: turn on/off excessive parameters monitoring
        """
        print(self.model)
        self.monitor.log_model(self.model)
        self.log_trainer()
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.model.cuda()
        if save:
            best_accuracy = self.load_best_accuracy()
        else:
            best_accuracy = 0.
        self.monitor.log(f"Best train accuracy so far: {best_accuracy:.4f}")
        print(f"Training '{self.model.__class__.__name__}'. "
              f"Best {self.dataset_name} train accuracy so far: {best_accuracy:.4f}")

        eval_loader = torch.utils.data.DataLoader(dataset=self.train_loader.dataset,
                                                  batch_size=self.train_loader.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.train_loader.num_workers)

        if with_mutual_info:
            global get_outputs
            get_outputs = self.monitor.mutual_info.decorate_evaluation(get_outputs)
            self.monitor.mutual_info.prepare(eval_loader)
        self.monitor.set_watch_mode(watch_parameters)

        for epoch in range(n_epoch):
            labels, outputs, loss = None, None, None
            for images, labels in tqdm(self.train_loader,
                                       desc="Epoch {:d}/{:d}".format(epoch, n_epoch),
                                       leave=False):
                if use_cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                outputs, loss = self.train_batch(images, labels)
                for name, param in self.model.named_parameters():
                    if torch.isnan(param).any():
                        warnings.warn(f"NaN parameters in '{name}'")
                self.monitor.batch_finished(self.model)

                # uncomment to see more detailed progress - at each batch instead of epoch
                # self.monitor.update_loss(loss=loss.item(), mode='batch')
                # self.monitor.update_accuracy(argmax_accuracy(outputs, labels), mode='batch')

            if epoch % epoch_update_step == 0:
                self.monitor.update_loss(loss=loss.item(), mode='batch')
                # self.monitor.update_accuracy(argmax_accuracy(outputs, labels), mode='batch')
                outputs_full, labels_full = get_outputs(self.model, eval_loader)
                # accuracy = argmax_accuracy(outputs_full, labels_full)
                # self.monitor.update_accuracy(accuracy, mode='full train')
                # if accuracy > best_accuracy:
                #     if save:
                #         self.save_model(accuracy)
                #     best_accuracy = accuracy
                #     self.monitor.log(f"Epoch {epoch}. Best train accuracy so far: {best_accuracy:.4f}")

                self.monitor.epoch_finished(self.model)
                self._epoch_finished(epoch, outputs_full, labels_full)
