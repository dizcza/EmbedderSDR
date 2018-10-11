import time
import warnings
from abc import ABC, abstractmethod
from functools import partial, update_wrapper

import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm

from loss import ContrastiveLoss
from monitor.accuracy import get_outputs, AccuracyEmbedding, AccuracyArgmax
from monitor.batch_timer import timer
from monitor.monitor import Monitor
from trainer.mask import MaskTrainer
from utils.common import get_data_loader, AdversarialExamples
from utils.constants import CHECKPOINTS_DIR
from utils.layers import find_named_layers
from utils.prepare import prepare_eval


class Trainer(ABC):
    watch_modules = (nn.Linear, nn.Conv2d)

    def __init__(self, model: nn.Module, criterion: nn.Module, dataset_name: str, env_suffix=''):
        self.model = model
        self.criterion = criterion
        self.dataset_name = dataset_name
        self.train_loader = get_data_loader(dataset_name, train=True)
        self.timer = timer
        self.timer.init(batches_in_epoch=len(self.train_loader))
        self.env_name = f"{time.strftime('%Y.%m.%d')} {self.model.__class__.__name__}: " \
                        f"{self.dataset_name} {self.__class__.__name__}"
        if env_suffix:
            self.env_name = self.env_name + f' {env_suffix}'
        if isinstance(self.criterion, ContrastiveLoss):
            self.accuracy_measure = AccuracyEmbedding()
        else:
            self.accuracy_measure = AccuracyArgmax()
        self.monitor = Monitor(test_loader=get_data_loader(self.dataset_name, train=False),
                               accuracy_measure=self.accuracy_measure)
        for name, layer in find_named_layers(self.model, layer_class=self.watch_modules):
            self.monitor.register_layer(layer, prefix=name)
        images, labels = next(iter(self.train_loader))
        self.mask_trainer = MaskTrainer(accuracy_measure=self.accuracy_measure, image_shape=images[0].shape)

    @property
    def checkpoint_path(self):
        return CHECKPOINTS_DIR / (self.env_name + '.pt')

    def monitor_functions(self):
        pass

    def log_trainer(self):
        self.monitor.log(f"Criterion: {self.criterion}")
        self.monitor.log(repr(self.mask_trainer))

    @abstractmethod
    def train_batch(self, images, labels):
        raise NotImplementedError()

    def save(self):
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), self.checkpoint_path)

    def state_dict(self):
        return {
            "model_state": self.model.state_dict(),
            "epoch": self.timer.epoch,
            "env_name": self.env_name,
        }

    def restore(self, checkpoint_path=None):
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path
        if not checkpoint_path.exists():
            print(f"Checkpoint '{checkpoint_path}' doesn't exist. Nothing to restore.")
            return None
        checkpoint_state = torch.load(checkpoint_path)
        try:
            self.model.load_state_dict(checkpoint_state['model_state'])
        except RuntimeError as error:
            print(f"Error is occurred while restoring {checkpoint_path}: {error}")
            return None
        self.env_name = checkpoint_state['env_name']
        self.timer.set_epoch(checkpoint_state['epoch'])
        self.monitor.open(env_name=self.env_name)
        print(f"Restored model state from {checkpoint_path}.")
        return checkpoint_state

    def _epoch_finished(self, epoch, outputs, labels):
        loss = self.criterion(outputs, labels)
        self.monitor.update_loss(loss, mode='full train')
        self.save()
        return loss

    def train_mask(self):
        """
        Train mask to see what part of the image is crucial from the network perspective.
        """
        images, labels = next(iter(self.train_loader))
        mode_saved = prepare_eval(self.model)
        if torch.cuda.is_available():
            images = images.cuda()
        with torch.no_grad():
            proba = self.accuracy_measure.predict_proba(self.model(images))
        proba_max, _ = proba.max(dim=1)
        sample_max_proba = proba_max.argmax()
        image = images[sample_max_proba]
        label = labels[sample_max_proba]
        self.monitor.plot_mask(self.model, mask_trainer=self.mask_trainer, image=image, label=label)
        mode_saved.restore(self.model)
        return image, label

    def get_adversarial_examples(self, noise_ampl=100, n_iter=10):
        """
        :param noise_ampl: adversarial noise amplitude
        :param n_iter: adversarial iterations
        :return adversarial examples
        """
        images, labels = next(iter(self.train_loader))
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()
        images_orig = images.clone()
        images.requires_grad_(True)
        mode_saved = prepare_eval(self.model)
        for i in range(n_iter):
            images.grad = None
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            with torch.no_grad():
                adv_noise = noise_ampl * images.grad
                images += adv_noise
        images.requires_grad_(False)
        mode_saved.restore(self.model)
        return AdversarialExamples(original=images_orig, adversarial=images, labels=labels)

    def train_epoch(self, epoch):
        """
        :param epoch: epoch id
        :return: last batch loss
        """
        loss = None
        use_cuda = torch.cuda.is_available()
        for images, labels in tqdm(self.train_loader,
                                   desc="Epoch {:d}".format(epoch),
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
            # self.monitor.activations_heatmap(outputs, labels)
            # self.monitor.update_loss(loss=loss.item(), mode='batch')

        return loss

    def train(self, n_epoch=10, epoch_update_step=1, watch_parameters=False,
              mutual_info_layers=1, adversarial=False, mask_explain=False):
        """
        :param n_epoch: number of training epochs
        :param epoch_update_step: epoch step to run full evaluation
        :param watch_parameters: turn on/off excessive parameters monitoring
        :param mutual_info_layers: number of last layers to be monitored for mutual information;
                                   pass '0' to turn off this feature.
        :param adversarial: perform adversarial attack test?
        :param mask_explain: train the image mask that 'explains' network behaviour?
        """
        print(self.model)
        if not self.monitor.is_active:
            # new environment
            self.monitor.open(env_name=self.env_name)
            self.monitor.clear()
        self.monitor_functions()
        self.monitor.log_model(self.model)
        self.log_trainer()
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.model.cuda()
        print(f"Training '{self.model.__class__.__name__}'")

        eval_loader = torch.utils.data.DataLoader(dataset=self.train_loader.dataset,
                                                  batch_size=self.train_loader.batch_size,
                                                  shuffle=False,
                                                  num_workers=self.train_loader.num_workers)

        get_outputs_eval = partial(get_outputs, loader=eval_loader)
        update_wrapper(wrapper=get_outputs_eval, wrapped=get_outputs)

        if mutual_info_layers > 0:
            get_outputs_eval = self.monitor.mutual_info.decorate_evaluation(get_outputs_eval)
            self.monitor.mutual_info.prepare(eval_loader, model=self.model, monitor_layers_count=mutual_info_layers)
        self.monitor.set_watch_mode(watch_parameters)

        for epoch in range(self.timer.epoch, self.timer.epoch + n_epoch):
            loss_batch = self.train_epoch(epoch=epoch)
            if epoch % epoch_update_step == 0:
                self.monitor.update_loss(loss=loss_batch, mode='batch')
                outputs_full, labels_full = get_outputs_eval(self.model)
                self.accuracy_measure.save(outputs_train=outputs_full, labels_train=labels_full)
                self.monitor.epoch_finished(self.model, outputs_full, labels_full)
                if adversarial:
                    self.monitor.plot_adversarial_examples(self.model, self.get_adversarial_examples())
                if mask_explain:
                    self.train_mask()
                self._epoch_finished(epoch, outputs_full, labels_full)
