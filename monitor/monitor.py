from collections import UserDict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from sklearn.metrics.pairwise import manhattan_distances

from monitor.accuracy import calc_accuracy
from monitor.batch_timer import timer, Schedule
from monitor.mutual_info import MutualInfoKMeans
from monitor.var_online import VarianceOnline
from monitor.viz import VisdomMighty
from utils import factors_root


class ParamRecord(object):
    def __init__(self, param: nn.Parameter):
        self.param = param
        self.is_monitored = False
        self.variance = None
        self.grad_variance = None
        self.prev_sign = None
        self.initial_data = None
        self.initial_norm = None

    def set_watch_mode(self, mode: bool):
        self.is_monitored = mode
        if self.is_monitored:
            data_cpu = self.param.data.cpu()
            self.variance = VarianceOnline(tensor=data_cpu)
            self.grad_variance = VarianceOnline()
            self.prev_sign = data_cpu.clone()  # clone is faster
            self.initial_data = data_cpu.clone()
            self.initial_norm = self.initial_data.norm(p=2)
        else:
            self.variance = None
            self.grad_variance = None
            self.prev_sign = None
            self.initial_data = None
            self.initial_norm = None

    def tstat(self) -> torch.FloatTensor:
        """
        :return: t-statistics of the parameters history
        """
        assert self.is_monitored, "Parameter is not monitored!"
        mean, std = self.variance.get_mean_std()
        tstat = mean.abs() / std
        isnan = std == 0
        if isnan.all():
            tstat.fill_(0)
        else:
            tstat_nonnan = tstat[~isnan]
            tstat_max = tstat_nonnan.mean() + 2 * tstat_nonnan.std()
            tstat_nonnan.clamp_(max=tstat_max)
            tstat[~isnan] = tstat_nonnan
            tstat[isnan] = tstat_max
        return tstat


class ParamsDict(UserDict):
    def __init__(self):
        super().__init__()
        self.sign_flips = 0
        self.n_updates = 0

    def batch_finished(self):
        self.n_updates += 1
        for param_record in self.values_monitored():
            param = param_record.param
            new_data = param.data.cpu()
            if new_data is param.data:
                new_data = new_data.clone()
            self.sign_flips += torch.sum((new_data * param_record.prev_sign) < 0)
            param_record.prev_sign = new_data
            param_record.variance.update(new_data)

    def plot_sign_flips(self, viz: VisdomMighty):
        if self.count_monitored() == 0:
            # haven't registered any monitored params yet
            return
        viz.line_update(y=self.sign_flips / self.n_updates, opts=dict(
            xlabel='Epoch',
            ylabel='Sign flips',
            title="Sign flips after optimizer.step()",
        ))
        self.sign_flips = 0
        self.n_updates = 0

    def items_monitored(self):
        def pass_monitored(pair):
            name, param_record = pair
            return param_record.is_monitored

        return filter(pass_monitored, self.items())

    def items_monitored_dict(self):
        return {name: param for name, param in self.items_monitored()}

    def values_monitored(self):
        for name, param_record in self.items_monitored():
            yield param_record

    def count_monitored(self):
        return len(list(self.values_monitored()))


class Monitor(object):

    n_classes_format_ytickstep_1 = 10

    def __init__(self, test_loader: torch.utils.data.DataLoader, env_name="main"):
        """
        :param test_loader: dataloader to test model performance at each epoch_finished() call
        :param env_name: Visdom environment name
        """
        self.timer = timer
        self.viz = VisdomMighty(env=env_name, timer=self.timer)
        self.test_loader = test_loader
        self.param_records = ParamsDict()
        self.mutual_info = MutualInfoKMeans(estimate_size=int(1e3), compression_range=(0.5, 0.999))
        self.functions = []

    def log_model(self, model: nn.Module, space='-'):
        for line in repr(model).splitlines():
            n_spaces = len(line) - len(line.lstrip())
            line = space * n_spaces + line
            self.viz.text(line, win='log', append=self.viz.win_exists('log'))

    def log(self, text: str):
        self.viz.log(text)

    def batch_finished(self, model: nn.Module):
        self.param_records.batch_finished()
        self.timer.tick()
        if self.timer.epoch == 0:
            self.mutual_info.force_update(model)
            self.mutual_info.plot(self.viz)

    def update_loss(self, loss: float, mode='batch'):
        self.viz.line_update(loss, opts=dict(
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

    @Schedule(epoch_update=1)
    def update_accuracy_test(self, model: nn.Module):
        self.update_accuracy(accuracy=calc_accuracy(model, self.test_loader), mode='full test')

    def register_func(self, func: Callable, opts: dict = None):
        self.functions.append((func, opts))

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
        for name, param_record in self.param_records.items_monitored():
            param = param_record.param
            if param.grad is None:
                continue
            param_record.grad_variance.update(param.grad.data.cpu())
            mean, std = param_record.grad_variance.get_mean_std()
            param_norm = param.data.norm(p=2)
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

    def epoch_finished(self, model: nn.Module):
        # self.update_accuracy_test(model)
        # self.update_distribution()
        self.mutual_info.plot(self.viz)
        for func_id, (func, opts) in enumerate(self.functions):
            self.viz.line_update(y=func(), opts=opts)
        # statistics below require monitored parameters
        # self.param_records.plot_sign_flips(self.viz)
        # self.update_gradient_mean_std()
        self.update_initial_difference()
        self.update_grad_norm()
        # self.update_heatmap_history(model, by_dim=False)

    def register_layer(self, layer: nn.Module, prefix: str):
        self.mutual_info.register(layer, name=prefix)
        for name, param in layer.named_parameters(prefix=prefix):
            self.param_records[name] = ParamRecord(param)

    def update_initial_difference(self):
        legend = []
        dp_normed = []
        for name, param_record in self.param_records.items_monitored():
            legend.append(name)
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
        for name, param_record in self.param_records.items_monitored():
            grad = param_record.param.grad
            if grad is not None:
                grad_norms.append(grad.data.norm(p=2))
                legend.append(name)
        if len(grad_norms) > 0:
            self.viz.line_update(y=grad_norms, opts=dict(
                xlabel='Epoch',
                ylabel='Gradient norm, L2',
                title='Gradient norm',
                legend=legend,
            ))

    def update_heatmap_history(self, model: nn.Module, by_dim=False):
        """
        :param model: current model
        :param by_dim: use hitmap_by_dim for the last layer's weights
        """
        def heatmap(tensor: torch.FloatTensor, win: str):
            while tensor.dim() > 2:
                tensor = tensor.mean(dim=0)
            opts = dict(
                colormap='Jet',
                title=win,
                xlabel='input dimension',
                ylabel='output dimension',
            )
            if tensor.shape[0] <= self.n_classes_format_ytickstep_1:
                opts.update(ytickstep=1)
            self.viz.heatmap(X=tensor, win=win, opts=opts)

        def heatmap_by_dim(tensor: torch.FloatTensor, win: str):
            for dim, x_dim in enumerate(tensor):
                factors = factors_root(x_dim.shape[0])
                x_dim = x_dim.view(factors)
                heatmap(x_dim, win=f'{win}: dim {dim}')

        names_backward = list(name for name, _ in model.named_parameters())[::-1]
        name_last = None
        for name in names_backward:
            if name in self.param_records:
                name_last = name
                break

        for name, param_record in self.param_records.items_monitored():
            heatmap_func = heatmap_by_dim if by_dim and name == name_last else heatmap
            heatmap_func(tensor=param_record.tstat(), win=f'Heatmap {name} t-statistics')

    def set_watch_mode(self, mode=False):
        for param_record in self.param_records.values():
            param_record.set_watch_mode(mode)

    def activations_heatmap(self, outputs: torch.Tensor, labels: torch.Tensor):
        """
        We'd like the last layer activations heatmap to be different for each corresponding label.
        :param outputs: the last layer activations
        :param labels: corresponding labels
        """
        outputs = outputs.detach()
        outputs_mean = []
        label_names = []
        for label in sorted(labels.unique()):
            outputs_mean.append(outputs[labels == label].mean(dim=0))
            label_names.append(str(label.item()))
        win = "Last layer activations heatmap"
        outputs_mean = torch.stack(outputs_mean, dim=0)
        opts = dict(
            title=win,
            xlabel='Embedding dimension',
            ylabel='Label',
            rownames=label_names,
        )
        if outputs_mean.shape[0] <= self.n_classes_format_ytickstep_1:
            opts.update(ytickstep=1)
        self.viz.heatmap(outputs_mean, win=win, opts=opts)
        l1_dist = manhattan_distances(outputs_mean)
        upper_triangle_idx = np.triu_indices_from(l1_dist, k=1)
        l1_dist = l1_dist[upper_triangle_idx].mean()
        self.viz.line_update(y=l1_dist, opts=dict(
            xlabel='Epoch',
            ylabel='Mean pairwise distance',
            title='How much do patterns differ in L1 measure?',
        ))
