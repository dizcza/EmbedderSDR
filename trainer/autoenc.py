import warnings

import torch
import torch.utils.data
from mighty.monitor.accuracy import how_many_samples_take
from mighty.monitor.mutual_info import MutualInfoStub
from mighty.monitor.var_online import MeanOnline
from trainer.kwta import TrainerGradKWTA
from mighty.utils.common import find_named_layers
from mighty.utils.data import get_normalize_inverse
from tqdm import tqdm

from monitor.monitor import MonitorAutoenc


class TrainerAutoenc(TrainerGradKWTA):

    def _init_monitor(self, mutual_info):
        normalize_inverse = get_normalize_inverse(self.data_loader.normalize)
        monitor = MonitorAutoenc(
            accuracy_measure=self.accuracy_measure,
            mutual_info=mutual_info,
            normalize_inverse=normalize_inverse
        )
        return monitor

    def forward_pass(self):
        eval_loader = self.data_loader.eval
        n_samples_take = how_many_samples_take(eval_loader)
        mode_saved = self.model.training
        self.model.train(False)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.model.cuda()
        loss_online = MeanOnline()
        with torch.no_grad():
            for bid, (inputs, labels) in enumerate(iter(eval_loader)):
                if eval_loader.batch_size * bid > n_samples_take:
                    break
                if use_cuda:
                    inputs = inputs.cuda()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, inputs)
                loss_online.update(loss)
        self.model.train(mode_saved)
        loss = loss_online.get_mean()
        return loss

    def train_batch(self, images, labels):
        self.optimizer.zero_grad()
        outputs = self.model(images)
        loss = self.criterion(outputs, images)
        loss.backward()
        self.optimizer.step(closure=None)
        return outputs, loss

    def train_epoch(self, epoch):
        """
        :param epoch: epoch id
        :return: last batch loss
        """
        loss_online = MeanOnline()
        use_cuda = torch.cuda.is_available()
        for images, labels in tqdm(self.train_loader,
                                   desc="Epoch {:d}".format(epoch),
                                   leave=False):
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()

            outputs, loss = self.train_batch(images, labels)
            loss_online.update(loss.detach().cpu())
            for name, param in self.model.named_parameters():
                if torch.isnan(param).any():
                    warnings.warn(f"NaN parameters in '{name}'")
            self.monitor.batch_finished(self.model)

        self.monitor.update_loss(loss=loss_online.get_mean(),
                                 mode='batch')

    def plot_autoencoder(self):
        images, _ = next(iter(self.data_loader.eval))
        self.model.eval()
        with torch.no_grad():
            images_reconstructed = self.model(images)
        self.monitor.plot_autoencoder(images, images_reconstructed)
        self.model.train()

    def train(self, n_epoch=10, epoch_update_step=1, mutual_info_layers=1,
              adversarial=False, mask_explain=False):
        """
        :param n_epoch: number of training epochs
        :param epoch_update_step: epoch step to run full evaluation
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
        self.monitor.log_self()
        self.log_trainer()
        for name, layer in find_named_layers(self.model,
                                             layer_class=self.watch_modules):
            self.monitor.register_layer(layer, prefix=name)

        eval_loader = self.data_loader.eval

        if mutual_info_layers > 0 and not isinstance(self.mutual_info,
                                                     MutualInfoStub):
            self.forward_pass = self.mutual_info.decorate_evaluation(
                self.forward_pass)
            self.mutual_info.prepare(eval_loader, model=self.model,
                                     monitor_layers_count=mutual_info_layers)

        print(f"Training '{self.model.__class__.__name__}'")

        for epoch in range(self.timer.epoch, self.timer.epoch + n_epoch):
            self.train_epoch(epoch=epoch)
            if epoch % epoch_update_step == 0:
                loss_full = self.forward_pass()
                self.monitor.epoch_finished()
                self.plot_autoencoder()
                if adversarial:
                    self.monitor.plot_adversarial_examples(
                        self.model,
                        self.get_adversarial_examples())
                if mask_explain:
                    self.train_mask()
                self.monitor.update_loss(loss_full, mode='full train')
                self.save()
