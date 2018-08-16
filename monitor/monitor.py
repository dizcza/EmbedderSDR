import time
from collections import UserDict
from typing import Callable

import torch
import torch.nn as nn
import torch.utils.data

from monitor.accuracy import calc_accuracy
from monitor.batch_timer import timer, Schedule
from monitor.mutual_info import MutualInfoKMeans
from monitor.var_online import VarianceOnline
from monitor.viz import VisdomMighty
from utils import get_data_loader


def timer_profile(func):
    def wrapped(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        elapsed = time.time() - start
        elapsed /= len(args[1])  # fps
        elapsed *= 1e3
        print(f"{func.__name__} {elapsed} ms")
        return res

    return wrapped


class ParamRecord(object):
    def __init__(self, param: nn.Parameter, monitor=False):
        self.param = param
        self.is_monitored = monitor
        self.variance = VarianceOnline(tensor=param.data.cpu(), is_active=self.is_monitored)
        self.grad_variance = VarianceOnline(is_active=self.is_monitored)
        if self.is_monitored:
            self.prev_sign = param.data.cpu().clone()  # clone is faster
            self.initial_data = param.data.clone()
            self.initial_norm = self.initial_data.norm(p=2)

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
        viz.line_update(y=self.sign_flips / self.n_updates, win='sign', opts=dict(
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

    def __init__(self, trainer, watch_parameters=False):
        """
        :param trainer: Trainer instance
        """
        self.watch_parameters = watch_parameters
        self.timer = timer
        self.timer.init(batches_in_epoch=len(trainer.train_loader))
        self.viz = VisdomMighty(env=f"{time.strftime('%Y-%b-%d')} "
                                    f"{trainer.dataset_name} "
                                    f"{trainer.__class__.__name__}", timer=self.timer)
        self.test_loader = get_data_loader(dataset=trainer.dataset_name, train=False)
        self.param_records = ParamsDict()
        self.mutual_info = MutualInfoKMeans(estimate_size=int(1e3), compression_range=(0.5, 0.999))
        self.functions = []
        self.log_model(trainer.model)
        self.log_trainer(trainer)

    def log_trainer(self, trainer):
        self.log(f"Criterion: {trainer.criterion}")
        optimizer = getattr(trainer, 'optimizer', None)
        if optimizer is not None:
            optimizer_str = f"Optimizer {optimizer.__class__.__name__}:"
            for group_id, group in enumerate(optimizer.param_groups):
                optimizer_str += f"\n\tgroup {group_id}: lr={group['lr']}, weight_decay={group['weight_decay']}"
            self.log(optimizer_str)

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
            self.mutual_info.update(model)
            self.mutual_info.plot(self.viz)

    def start_training(self, model: nn.Module):
        self.mutual_info.update(model)
        self.mutual_info.plot(self.viz)

    def update_loss(self, loss: float, mode='batch'):
        self.viz.line_update(loss, win=f'loss', opts=dict(
            xlabel='Epoch',
            ylabel='Loss',
            title=f'Loss'
        ), name=mode)

    def update_accuracy(self, accuracy: float, mode='batch'):
        self.viz.line_update(accuracy, win=f'accuracy', opts=dict(
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
            param = param_record.param
            if param.numel() == 1:
                self.viz.line_update(y=param.data[0], win=name, opts=dict(
                    xlabel='Epoch',
                    ylabel='Value',
                    title=name,
                ))
            else:
                self.viz.histogram(X=param.data.cpu().view(-1), win=name, opts=dict(
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
            self.viz.line_update(y=[mean, std], win=f"grad_mean_std_{name}", opts=dict(
                xlabel='Epoch',
                ylabel='Normalized Mean and STD',
                title=name,
                legend=['||Mean(∇Wi)||', 'STD(∇Wi)'],
                xtype='log',
                ytype='log',
            ))

    def epoch_finished(self, model: nn.Module):
        # self.update_accuracy_test(model)
        self.update_distribution()
        self.mutual_info.plot(self.viz)
        for func_id, (func, opts) in enumerate(self.functions):
            self.viz.line_update(y=func(), win=f"func_{func_id}", opts=opts)
        # statistics below require monitored parameters
        self.param_records.plot_sign_flips(self.viz)
        self.update_gradient_mean_std()
        self.update_initial_difference()
        self.update_grad_norm()

    def register_layer(self, layer: nn.Module, prefix: str):
        self.mutual_info.register(layer, name=prefix)
        for name, param in layer.named_parameters(prefix=prefix):
            self.param_records[name] = ParamRecord(param, monitor=self.watch_parameters)

    def update_initial_difference(self):
        legend = []
        dp_normed = []
        for name, param_record in self.param_records.items_monitored():
            legend.append(name)
            dp = param_record.param.data - param_record.initial_data
            dp = dp.norm(p=2) / param_record.initial_norm
            dp_normed.append(dp)
        self.viz.line_update(y=dp_normed, win='w_initial', opts=dict(
            xlabel='Epoch',
            ylabel='||W - W_initial|| / ||W_initial||',
            title='How far the current weights are from the initial?',
            legend=legend,
        ))

    def update_grad_norm(self):
        grad_norms = []
        for name, param_record in self.param_records.items_monitored():
            grad = param_record.param.grad
            if grad is not None:
                grad_norms.append(grad.data.norm(p=2))
        if len(grad_norms) > 0:
            norm_mean = sum(grad_norms) / len(grad_norms)
            self.viz.line_update(y=norm_mean, win='grad_norm', opts=dict(
                xlabel='Epoch',
                ylabel='Gradient norm, L2',
                title='Average grad norm of all params',
            ))
