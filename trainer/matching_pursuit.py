import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data

from mighty.monitor.var_online import MeanOnline
from mighty.trainer.trainer import Trainer
from mighty.utils.algebra import compute_psnr, compute_sparsity
from mighty.utils.data import DataLoader
from mighty.utils.stub import OptimizerStub
from .autoencoder import TrainerAutoencoderBinary


class TestMatchingPursuitParameters(TrainerAutoencoderBinary):

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 data_loader: DataLoader,
                 bmp_params_range: torch.Tensor,
                 param_name="param",
                 **kwargs):
        super().__init__(model,
                         criterion=criterion,
                         data_loader=data_loader,
                         optimizer=OptimizerStub(),
                         **kwargs)
        self.bmp_params = bmp_params_range
        self.param_name = param_name

    def train_epoch(self, epoch):
        self.timer.batch_id += self.timer.batches_in_epoch

    def full_forward_pass(self, train=True):
        if not train:
            return None
        assert isinstance(self.criterion,
                          nn.MSELoss), "BMP can work only with MSE loss"

        mode_saved = self.model.training
        self.model.train(False)
        use_cuda = torch.cuda.is_available()

        loss_online = MeanOnline()
        psnr_online = MeanOnline()
        sparsity_online = MeanOnline()
        with torch.no_grad():
            for inputs, labels in self.eval_batches(verbose=True):
                if use_cuda:
                    inputs = inputs.cuda()
                loss = []
                psnr = []
                sparsity = []
                for bmp_param in self.bmp_params:
                    outputs = self.model(inputs, bmp_param)
                    latent, reconstructed = outputs
                    loss_lambd = self._get_loss(inputs, outputs, labels)
                    psnr_lmdb = compute_psnr(inputs, reconstructed)
                    sparsity_lambd = compute_sparsity(latent)
                    loss.append(loss_lambd.cpu())
                    psnr.append(psnr_lmdb.cpu())
                    sparsity.append(sparsity_lambd.cpu())

                loss_online.update(torch.stack(loss))
                psnr_online.update(torch.stack(psnr))
                sparsity_online.update(torch.stack(sparsity))

        loss = loss_online.get_mean()
        self.monitor.viz.line(Y=loss, X=self.bmp_params, win='Loss',
                              opts=dict(
                                  xlabel=f'BMP {self.param_name}',
                                  ylabel='Loss',
                                  title='Loss'
                              ))

        psnr = psnr_online.get_mean()
        self.monitor.viz.line(Y=psnr, X=self.bmp_params, win='PSNR',
                              opts=dict(
                                  xlabel=f'BMP {self.param_name}',
                                  ylabel='Peak signal-to-noise ratio',
                                  title='PSNR'
                              ))

        sparsity = sparsity_online.get_mean()
        self.monitor.viz.line(Y=sparsity, X=self.bmp_params, win='Sparsity',
                              opts=dict(
                                  xlabel=f'BMP {self.param_name}',
                                  ylabel='sparsity',
                                  title='L1 output sparsity'
                              ))

        self.monitor.viz.close(win='Accuracy')
        self.model.train(mode_saved)

        return loss

    def _epoch_finished(self, epoch, loss):
        Trainer._epoch_finished(self, epoch, loss)


class TestMatchingPursuit(TrainerAutoencoderBinary):

    def train_epoch(self, epoch):
        self.timer.batch_id += self.timer.batches_in_epoch
