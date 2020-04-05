import torch
import torch.utils.data
import torch.utils.data

from mighty.monitor.var_online import MeanOnline
from mighty.utils.algebra import compute_psnr
from .autoencoder import TrainerAutoencoderBinary


class TrainerLISTA(TrainerAutoencoderBinary):

    def _init_online_measures(self):
        online = super()._init_online_measures()
        online['psnr-bmp'] = MeanOnline()
        return online

    def _get_loss(self, input, output, labels):
        encoded, decoded, bmp_encoded, bmp_decoded = output
        return self.criterion(encoded, bmp_encoded)

    def _on_forward_pass_batch(self, input, output, labels):
        encoded, decoded, bmp_encoded, bmp_decoded = output
        super()._on_forward_pass_batch(input,
                                       output=(encoded, decoded),
                                       labels=labels)
        psnr = compute_psnr(input, bmp_decoded)
        self.online['psnr-bmp'].update(psnr.cpu())

    def _epoch_finished(self, epoch, loss):
        super()._epoch_finished(epoch, loss)
        self.monitor.plot_psnr(self.online['psnr-bmp'].get_mean(), win='BMP')

    def plot_autoencoder(self):
        input, labels = next(iter(self.data_loader.eval))
        if torch.cuda.is_available():
            input = input.cuda()
        mode_saved = self.model.training
        self.model.train(False)
        with torch.no_grad():
            latent, reconstructed, _, _ = self.model(input)

        lowest_id = self.online['pixel-error'].get_mean().argmin()
        thr_lowest = self.reconstruct_thr[0, 0, lowest_id]
        rec_binary = (reconstructed >= thr_lowest).type(torch.float32)
        self.monitor.plot_autoencoder_binary(input, reconstructed, rec_binary)
        self.model.train(mode_saved)
