import os
import subprocess
from collections import UserDict
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import confusion_matrix, pairwise

from monitor.accuracy import calc_accuracy, full_forward_pass, Accuracy
from monitor.batch_timer import timer, ScheduleStep
from monitor.mutual_info import MutualInfoKMeans
from monitor.var_online import VarianceOnline
from monitor.viz import VisdomMighty
from utils.common import AdversarialExamples
from utils.normalize import get_normalize_inverse


class ParamRecord(object):
    def __init__(self, param: nn.Parameter):
        self.param = param
        self.is_monitored = False
        self.grad_variance = VarianceOnline()
        self.prev_sign = None
        self.initial_data = None
        self.initial_norm = param.data.norm(p=2).item()


class ParamsDict(UserDict):
    def __init__(self):
        super().__init__()
        self.sign_flips = 0
        self.n_updates = 0

    def batch_finished(self):
        self.n_updates += 1
        for param_record in filter(lambda precord: precord.prev_sign is not None, self.values()):
            param = param_record.param
            new_data = param.data.cpu()
            if new_data is param.data:
                new_data = new_data.clone()
            self.sign_flips += (new_data * param_record.prev_sign < 0).sum().item()
            param_record.prev_sign = new_data

    def plot_sign_flips(self, viz: VisdomMighty):
        for param_record in filter(lambda precord: precord.prev_sign is None, self.values()):
            param_record.prev_sign = param_record.param.data.cpu().clone()
        viz.line_update(y=self.sign_flips / self.n_updates, opts=dict(
            xlabel='Epoch',
            ylabel='Sign flips',
            title="Sign flips after optimizer.step()",
        ))
        self.sign_flips = 0
        self.n_updates = 0


class Monitor(object):
    n_classes_format_ytickstep_1 = 10

    def __init__(self, test_loader: torch.utils.data.DataLoader, accuracy_measure: Accuracy):
        """
        :param test_loader: dataloader to test model performance at each epoch_finished() call
        :param accuracy_measure: argmax or centroid embeddings accuracy measure
        """
        self.timer = timer
        self.test_loader = test_loader
        self.viz = None
        self.normalize_inverse = None
        self.advanced_monitoring = False  # memory consuming
        if self.test_loader is not None:
            self.normalize_inverse = get_normalize_inverse(self.test_loader.dataset.transform)
        self.accuracy_measure = accuracy_measure
        self.param_records = ParamsDict()
        self.mutual_info = MutualInfoKMeans(estimate_size=int(1e3), compression_range=(0.5, 0.999))
        self.functions = []

    @property
    def is_active(self):
        return self.viz is not None

    def open(self, env_name: str):
        """
        :param env_name: Visdom environment name
        """
        self.viz = VisdomMighty(env=env_name)
        self.viz.prepare()

    def log_model(self, model: nn.Module, space='-'):
        lines = []
        for line in repr(model).splitlines():
            n_spaces = len(line) - len(line.lstrip())
            line = space * n_spaces + line
            lines.append(line)
        lines = '<br>'.join(lines)
        self.log(lines)

    def log_self(self):
        self.log(f"{self.__class__.__name__}(accuracy_measure={self.accuracy_measure.__class__.__name__})")
        self.log(f"FULL_FORWARD_PASS_SIZE: {os.environ.get('FULL_FORWARD_PASS_SIZE', '(all samples)')}")
        self.log(f"BATCH_SIZE: {os.environ.get('BATCH_SIZE', '(default)')}")
        self.log(f"Batches in epoch: {self.timer.batches_in_epoch}")
        self.log(f"Start epoch: {self.timer.epoch}")
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE, universal_newlines=True)
        self.log(f"Git commit: {commit.stdout}")

    def log(self, text: str):
        self.viz.log(text)

    def batch_finished(self, model: nn.Module):
        self.param_records.batch_finished()
        self.timer.tick()
        if self.timer.epoch == 0:
            self.mutual_info.force_update(model)
            self.mutual_info.plot(self.viz)

    def update_loss(self, loss: Optional[torch.Tensor], mode='batch'):
        if loss is None:
            return
        self.viz.line_update(loss.item(), opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title=f'Loss'
        ), name=mode)

    def update_accuracy(self, accuracy: float, mode='batch'):
        self.viz.line_update(accuracy, opts=dict(
            xlabel='Epoch',
            ylabel='Accuracy',
            title=f'Accuracy'
        ), name=mode)

    def clear(self):
        self.viz.close()
        self.viz.prepare()

    def register_func(self, *func: Callable):
        self.functions.extend(func)

    def update_distribution(self):
        for name, param_record in self.param_records.items():
            param_data = param_record.param.data.cpu()
            if param_data.numel() == 1:
                self.viz.line_update(y=param_data.item(), opts=dict(
                    xlabel='Epoch',
                    ylabel='Value',
                    title=name,
                ))
            else:
                self.viz.histogram(X=param_data.view(-1), win=name, opts=dict(
                    xlabel='Param norm',
                    ylabel='# bins (distribution)',
                    title=name,
                ))

    def update_gradient_mean_std(self):
        for name, param_record in self.param_records.items():
            param = param_record.param
            if param.grad is None:
                continue
            param_record.grad_variance.update(param.grad.data.cpu())
            mean, std = param_record.grad_variance.get_mean_std()
            param_norm = param.data.norm(p=2).item()
            mean = mean.norm(p=2) / param_norm
            std = std.mean() / param_norm
            self.viz.line_update(y=[mean, std], opts=dict(
                xlabel='Epoch',
                ylabel='Normalized Mean and STD',
                title=f'Gradient Mean and STD: {name}',
                legend=['||Mean(∇Wi)||', 'STD(∇Wi)'],
                xtype='log',
                ytype='log',
            ))

    def plot_accuracy_confusion_matrix(self, labels_true, labels_predicted, mode):
        self.update_accuracy(accuracy=calc_accuracy(labels_true, labels_predicted), mode=mode)
        title = f"Confusion matrix '{mode}'"
        if len(labels_true.unique()) <= self.n_classes_format_ytickstep_1:
            # don't plot huge matrix
            confusion = confusion_matrix(labels_true, labels_predicted)
            self.viz.heatmap(confusion, win=title, opts=dict(
                title=title,
                xlabel='Predicted label',
                ylabel='True label',
            ))

    def update_accuracy_epoch(self, model: nn.Module, outputs_train, labels_train):
        outputs_test, labels_test = full_forward_pass(model, loader=self.test_loader)
        self.plot_accuracy_confusion_matrix(labels_true=labels_train,
                                            labels_predicted=self.accuracy_measure.predict(outputs_train),
                                            mode='full train')
        self.plot_accuracy_confusion_matrix(labels_true=labels_test,
                                            labels_predicted=self.accuracy_measure.predict(outputs_test),
                                            mode='full test')

    def plot_adversarial_examples(self, model: nn.Module, adversarial_examples: AdversarialExamples, n_show=10):
        images_orig, images_adv, labels_true = adversarial_examples
        saved_mode = model.training
        model.eval()
        with torch.no_grad():
            outputs_orig = model(images_orig)
            outputs_adv = model(images_adv)
        model.train(saved_mode)
        accuracy_orig = calc_accuracy(labels_true=labels_true,
                                      labels_predicted=self.accuracy_measure.predict(outputs_orig))
        accuracy_adv = calc_accuracy(labels_true=labels_true,
                                     labels_predicted=self.accuracy_measure.predict(outputs_adv))
        self.update_accuracy(accuracy=accuracy_orig, mode='batch')
        self.update_accuracy(accuracy=accuracy_adv, mode='adversarial')
        images_stacked = []
        images_orig, images_adv = images_orig.cpu(), images_adv.cpu()
        for images in (images_orig, images_adv):
            n_show = min(n_show, len(images))
            images = images[: n_show]
            if self.normalize_inverse is not None:
                images = list(map(self.normalize_inverse, images))
                images = torch.cat(images, dim=2)
            images_stacked.append(images)
        adv_noise = images_stacked[1] - images_stacked[0]
        adv_noise -= adv_noise.min()
        adv_noise /= adv_noise.max()
        images_stacked.insert(1, adv_noise)
        images_stacked = torch.cat(images_stacked, dim=1)
        images_stacked.clamp_(0, 1)
        self.viz.image(images_stacked, win='Adversarial examples', opts=dict(title='Adversarial examples'))

    def plot_mask(self, model: nn.Module, mask_trainer, image, label, win_suffix=''):
        def forward_probability(image_example):
            with torch.no_grad():
                outputs = model(image_example.unsqueeze(dim=0))
            proba = mask_trainer.get_probability(outputs=outputs, label=label)
            return proba
        mask, loss_trace, image_perturbed = mask_trainer.train_mask(model=model, image=image, label_true=label)
        proba_original = forward_probability(image)
        proba_perturbed = forward_probability(image_perturbed)
        image, mask, image_perturbed = image.cpu(), mask.cpu(), image_perturbed.cpu()
        image = self.normalize_inverse(image)
        image_perturbed = self.normalize_inverse(image_perturbed)
        image_masked = mask * image
        images_stacked = torch.stack([image, mask, image_masked, image_perturbed], dim=0)
        images_stacked.clamp_(0, 1)
        self.viz.images(images_stacked, nrow=len(images_stacked), win=f'masked images {win_suffix}', opts=dict(
            title=f"Masked image decreases neuron '{label}' probability {proba_original:.4f} -> {proba_perturbed:.4f}"
        ))
        self.viz.line(Y=loss_trace, X=np.arange(1, len(loss_trace)+1), win=f'mask loss {win_suffix}', opts=dict(
            xlabel='Iteration',
            title='Mask loss'
        ))

    def epoch_finished(self, model: nn.Module, outputs_full, labels_full):
        self.update_accuracy_epoch(model, outputs_train=outputs_full, labels_train=labels_full)
        # self.update_distribution()
        self.mutual_info.plot(self.viz)
        for monitored_function in self.functions:
            monitored_function(self.viz)
        self.update_grad_norm()
        self.update_sparsity(outputs_full, mode='full train')
        self.update_density(outputs_full, mode='full train')
        self.activations_heatmap(outputs_full, labels_full)
        self.firing_frequency(outputs_full)
        if self.advanced_monitoring:
            self.param_records.plot_sign_flips(self.viz)
            self.update_gradient_mean_std()
            self.update_initial_difference()

    def register_layer(self, layer: nn.Module, prefix: str):
        self.mutual_info.register(layer, name=prefix)
        for name, param in layer.named_parameters(prefix=prefix):
            if param.requires_grad:
                self.param_records[name] = ParamRecord(param)

    def update_sparsity(self, outputs, mode: str):
        outputs = outputs.detach()
        sparsity = outputs.norm(p=1, dim=1).mean() / outputs.shape[1]
        self.viz.line_update(y=sparsity.cpu(), opts=dict(
            xlabel='Epoch',
            ylabel='L1 norm / size',
            title='Output sparsity',
        ), name=mode)

    def update_density(self, outputs, mode: str):
        outputs = outputs.detach()
        mean = outputs.mean(dim=0)
        var = outputs.var(dim=0)
        density = 1 / (var / (mean * mean + 1e-7) + 1)
        density = density.mean()
        self.viz.line_update(y=density.cpu(), opts=dict(
            xlabel='Epoch',
            ylabel='(mean / std) ^ 2',
            title='Output density',
        ), name=mode)

    def update_initial_difference(self):
        legend = []
        dp_normed = []
        for name, param_record in self.param_records.items():
            legend.append(name)
            if param_record.initial_data is None:
                param_record.initial_data = param_record.param.data.cpu().clone()
            dp = param_record.param.data.cpu() - param_record.initial_data
            dp = dp.norm(p=2) / param_record.initial_norm
            dp_normed.append(dp)
        self.viz.line_update(y=dp_normed, opts=dict(
            xlabel='Epoch',
            ylabel='||W - W_initial|| / ||W_initial||',
            title='How far the current weights are from the initial?',
            legend=legend,
        ))

    def update_grad_norm(self):
        grad_norms = []
        legend = []
        for name, param_record in self.param_records.items():
            grad = param_record.param.grad
            if grad is not None:
                grad_norms.append(grad.norm(p=2).cpu())
                legend.append(name)
        if len(grad_norms) > 0:
            self.viz.line_update(y=grad_norms, opts=dict(
                xlabel='Epoch',
                ylabel='Gradient norm, L2',
                title='Gradient norm',
                legend=legend,
            ))

    def activations_heatmap(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        We'd like the last layer activations heatmap to be different for each corresponding label.
        :param outputs: the last layer activations
        :param labels: corresponding labels
        """

        def compute_manhattan_dist(tensor: torch.FloatTensor) -> float:
            l1_dist = pairwise.manhattan_distances(tensor.cpu())
            upper_triangle_idx = np.triu_indices_from(l1_dist, k=1)
            l1_dist = l1_dist[upper_triangle_idx].mean()
            return l1_dist

        outputs = outputs.detach()
        class_centroids = []
        std_centroids = []
        label_names = []
        for label in sorted(labels.unique()):
            outputs_label = outputs[labels == label]
            std_centroids.append(outputs_label.std(dim=0))
            class_centroids.append(outputs_label.mean(dim=0))
            label_names.append(str(label.item()))
        win = "Last layer activations heatmap"
        class_centroids = torch.stack(class_centroids, dim=0)
        std_centroids = torch.stack(std_centroids, dim=0)
        opts = dict(
            title=f"{win}. Epoch {self.timer.epoch}",
            xlabel='Embedding dimension',
            ylabel='Label',
            rownames=label_names,
        )
        if class_centroids.shape[0] <= self.n_classes_format_ytickstep_1:
            opts.update(ytickstep=1)
        self.viz.heatmap(class_centroids, win=win, opts=opts)
        self.save_heatmap(class_centroids, win=win, opts=opts)
        normalizer = class_centroids.norm(p=1, dim=1).mean()
        outer_distance = compute_manhattan_dist(class_centroids) / normalizer
        std = std_centroids.norm(p=1, dim=1).mean() / normalizer
        self.viz.line_update(y=[outer_distance.item(), std.item()], opts=dict(
            xlabel='Epoch',
            ylabel='Mean pairwise distance (normalized)',
            legend=['inter-distance', 'intra-STD'],
            title='How much do patterns differ in L1 measure?',
        ))

    @ScheduleStep(epoch_step=20)
    def save_heatmap(self, heatmap, win, opts):
        self.viz.heatmap(heatmap, win=f"{win}. Epoch {self.timer.epoch}", opts=opts)

    def firing_frequency(self, outputs):
        frequency = outputs.detach().mean(dim=0)
        frequency.unsqueeze_(dim=0)
        title = 'Neuron firing frequency'
        self.viz.heatmap(frequency, win=title, opts=dict(
            title=title,
            xlabel='Embedding dimension',
            rownames=['Last layer'],
        ))
