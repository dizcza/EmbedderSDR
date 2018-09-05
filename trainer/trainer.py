import time
import warnings
from abc import ABC, abstractmethod
from functools import partial, update_wrapper

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from constants import ROOT_DIR
from loss import ContrastiveLoss, ContrastiveLossBatch
from monitor.accuracy import get_outputs, calc_raw_accuracy
from monitor.batch_timer import timer
from monitor.monitor import Monitor
from trainer.checkpoint import Checkpoint
from utils import get_data_loader, create_pairs


class Trainer(ABC):

    watch_modules = (nn.Linear, nn.Conv2d)

    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str, patience=None):
        self.model = model
        self.criterion = criterion
        self.dataset_name = dataset_name
        self.train_loader = get_data_loader(dataset_name, train=True)
        timer.init(batches_in_epoch=len(self.train_loader))
        self.monitor = Monitor(test_loader=get_data_loader(self.dataset_name, train=False), env_name=self.env_name)
        self._monitor_parameters(self.model)
        self.checkpoint = Checkpoint(model=self.model, patience=patience)

    @property
    def env_name(self) -> str:
        env_name = f"{time.strftime('%Y.%m.%d')} {self.model.__class__.__name__}: " \
                   f"{self.dataset_name} {self.__class__.__name__}"
        env_name = env_name.replace('_', '-')  # visdom things
        return env_name

    def log_trainer(self):
        self.monitor.log(f"Criterion: {self.criterion}")

    def reset_checkpoint(self):
        self.checkpoint.reset(model=self.model)

    def _monitor_parameters(self, layer: nn.Module, prefix=''):
        for name, child in layer.named_children():
            self._monitor_parameters(child, prefix=f'{prefix}.{name}')
        if isinstance(layer, self.watch_modules):
            self.monitor.register_layer(layer, prefix=prefix.lstrip('.'))

    @abstractmethod
    def train_batch(self, images, labels):
        raise NotImplementedError()

    def train_batch_pairs(self, pairs_left, pairs_right, targets):
        raise NotImplementedError()

    def _epoch_finished(self, epoch, outputs, labels) -> torch.Tensor:
        if isinstance(self.criterion, ContrastiveLoss):
            criterion = self.criterion
        else:
            criterion = ContrastiveLossBatch(same_only=False)
        loss = criterion(outputs, labels).item()
        self.monitor.update_loss(loss, mode='full train')
        self.checkpoint.step(model=self.model, loss=loss)
        return loss

    def train(self, n_epoch=10, epoch_update_step=1, with_mutual_info=False, watch_parameters=False):
        """
        :param n_epoch: number of training epochs
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
        accuracy_message = f"Raw input centroids accuracy: {calc_raw_accuracy(self.train_loader):.3f}"
        self.monitor.log(accuracy_message)
        print(f"Training '{self.model.__class__.__name__}'. {accuracy_message}")

        eval_loader = torch.utils.data.DataLoader(dataset=self.train_loader.dataset,
                                                  batch_size=self.train_loader.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.train_loader.num_workers)

        get_outputs_eval = partial(get_outputs, loader=eval_loader)
        update_wrapper(wrapper=get_outputs_eval, wrapped=get_outputs)

        if with_mutual_info:
            get_outputs_eval = self.monitor.mutual_info.decorate_evaluation(get_outputs_eval)
            self.monitor.mutual_info.prepare(eval_loader, model=self.model, monitor_layers_count=5)
        self.monitor.set_watch_mode(watch_parameters)

        # do we need to transform images into pairs?
        as_pairs = not isinstance(self.criterion, ContrastiveLoss)

        for epoch in range(n_epoch):
            labels, outputs, loss = None, None, None
            for images, labels in tqdm(self.train_loader,
                                       desc="Epoch {:d}/{:d}".format(epoch, n_epoch),
                                       leave=False):
                if use_cuda:
                    images = images.cuda()
                    labels = labels.cuda()

                if as_pairs:
                    pairs_left, pairs_right, targets = create_pairs(images, labels)
                    outputs_left, outputs_right, loss = self.train_batch_pairs(pairs_left, pairs_right, targets)
                else:
                    outputs, loss = self.train_batch(images, labels)
                for name, param in self.model.named_parameters():
                    if torch.isnan(param).any():
                        warnings.warn(f"NaN parameters in '{name}'")
                self.monitor.batch_finished(self.model)

                # uncomment to see more detailed progress - at each batch instead of epoch
                # self.monitor.activations_heatmap(outputs, labels)
                # self.monitor.update_loss(loss=loss.item(), mode='batch')

            if epoch % epoch_update_step == 0:
                self.monitor.update_loss(loss=loss.item(), mode='batch')
                outputs_full, labels_full = get_outputs_eval(self.model)
                self.monitor.epoch_finished(self.model, outputs_full, labels_full)
                self._epoch_finished(epoch, outputs_full, labels_full)
