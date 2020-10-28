import numpy as np

from mighty.monitor.monitor import MonitorAutoencoder


class MonitorAutoencoderBinary(MonitorAutoencoder):

    def plot_autoencoder_binary(self, images, reconstructed,
                                reconstructed_binary, *tensors, labels=(),
                                normalize_inverse=True, n_show=10,
                                mode='train'):
        labels = ['Reconstructed binary', *labels]
        if normalize_inverse and self.normalize_inverse is not None:
            images = self.normalize_inverse(images)
            reconstructed = self.normalize_inverse(reconstructed)
            tensors = map(self.normalize_inverse, tensors)
            # reconstructed_binary is already in [0, 1] range
        self.plot_autoencoder(images, reconstructed, reconstructed_binary,
                              *tensors, labels=labels, normalize_inverse=False,
                              n_show=n_show, mode=mode)

    def plot_reconstruction_exact(self, n_exact, n_total=None, mode='train'):
        if n_total is not None:
            accuracy = n_exact / float(n_total)
            self.viz.line_update(accuracy, opts=dict(
                xlabel='Epoch',
                ylabel='Accuracy',
                title=f'Accuracy AutoEncoder'
            ), name=mode)

        named_metric = [(mode, n_exact)]
        if n_total is not None:
            named_metric.append((f"#total-{mode}", n_total))
        for name, val in named_metric:
            dash = 'solid' if 'total' not in name else 'dash'
            self.viz.line_update(val, opts=dict(
                title="Reconstruction exact",
                xlabel="Epoch",
                ylabel="num. of exactly reconstructed",
                dash=np.array([dash]),
            ), name=name)
