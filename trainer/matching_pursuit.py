import torch
import torch.nn as nn

from mighty.monitor.var_online import MeanOnline
from mighty.utils.algebra import compute_psnr, compute_sparsity
from trainer import TrainerAutoenc

from mighty.trainer.trainer import Trainer


class TestMatchingPursuit(TrainerAutoenc):
    bmp_lambdas = torch.linspace(0.05, 0.8, steps=20)

    def train_batch(self, images, labels):
        # never called
        return None, None

    def train_epoch(self, epoch):
        self.timer.batch_id += self.timer.batches_in_epoch

    def _on_forward_pass_batch(self, input, output, labels):
        latent, reconstructed = output
        psnr = compute_psnr(input, reconstructed)
        self.online['psnr'].update(psnr)

    def full_forward_pass(self):
        assert isinstance(self.criterion,
                          nn.MSELoss), "BMP can work only with MSE loss"

        mode_saved = self.model.training
        self.model.train(False)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.model.cuda()

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
                for lambd in self.bmp_lambdas:
                    outputs = self.model(inputs, lambd=lambd)
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
        self.monitor.viz.line(Y=loss, X=self.bmp_lambdas, win='Loss',
                              opts=dict(
                                  xlabel='BMP lambda',
                                  ylabel='Loss',
                                  title='Loss'
                              ))

        psnr = psnr_online.get_mean()
        self.monitor.viz.line(Y=psnr, X=self.bmp_lambdas, win='PSNR',
                              opts=dict(
                                  xlabel='BMP lambda',
                                  ylabel='Peak signal-to-noise ratio',
                                  title='PSNR'
                              ))

        sparsity = sparsity_online.get_mean()
        self.monitor.viz.line(Y=sparsity, X=self.bmp_lambdas, win='Sparsity',
                              opts=dict(
                                  xlabel='BMP lambda',
                                  ylabel='BMP sparsity',
                                  title='Sparsity'
                              ))

        self.monitor.viz.close(win='Accuracy')
        self.model.train(mode_saved)

        return loss

    def full_forward_pass_test(self):
        return 0.

    def _epoch_finished(self, epoch, loss):
        Trainer._epoch_finished(self, epoch, loss)
