from collections import defaultdict

import torch
import torch.utils.data
from tqdm import tqdm

from models.kwta import _KWinnersTakeAllFunction
from monitor.var_online import MeanOnline
from monitor.viz import VisdomMighty
from utils.algebra import factors_root
from utils.common import get_data_loader
from utils.normalize import get_normalize_inverse


def torch_to_matplotlib(image):
    if image.shape[0] == 1:
        return image.squeeze(dim=0)
    else:
        return image.transpose(0, 1).transpose(1, 2)


def undo_normalization(images_normalized, normalize_inverse):
    tensor = torch.stack(list(map(normalize_inverse, images_normalized)))
    tensor.clamp_(0, 1)
    return tensor


def kwta_inverse(embedding_dim=10000, sparsity=0.05, dataset="MNIST", debug=False):
    import matplotlib.pyplot as plt
    loader = get_data_loader(dataset=dataset, batch_size=32)
    normalize_inverse = get_normalize_inverse(loader.dataset.transform)
    images, labels = next(iter(loader))
    batch_size, channels, height, width = images.shape
    kwta_embeddings = []
    before_inverse = []
    restored = []
    for channel in range(channels):
        images_channel = images[:, channel, :, :]
        images_binary = (images_channel > 0).type(torch.float32)
        sparsity_channel = images_binary.mean()
        print(f"Sparsity image raw channel={channel}: {sparsity_channel:.3f}")
        images_flatten = images_channel.flatten(start_dim=1)
        weights = torch.randn(images_flatten.shape[1], embedding_dim)
        embeddings = images_flatten @ weights
        kwta_embeddings_channel = _KWinnersTakeAllFunction.apply(embeddings.clone(), sparsity)
        before_inverse_channel = kwta_embeddings_channel @ weights.transpose(0, 1)
        restored_channel = _KWinnersTakeAllFunction.apply(before_inverse_channel.clone(), sparsity_channel)
        kwta_embeddings.append(kwta_embeddings_channel)
        before_inverse.append(before_inverse_channel)
        restored.append(restored_channel)

    kwta_embeddings = torch.stack(kwta_embeddings, dim=1)
    before_inverse = torch.stack(before_inverse, dim=1)
    restored = torch.stack(restored, dim=1)

    kwta_embeddings = kwta_embeddings.view(batch_size, channels, *factors_root(embedding_dim))
    before_inverse = before_inverse.view_as(images)
    restored = restored.view_as(images)

    images = undo_normalization(images, normalize_inverse)

    for orig, kwta, raw, restored in zip(images, kwta_embeddings, before_inverse, restored):
        plt.subplot(141)
        plt.title("Original")
        plt.imshow(torch_to_matplotlib(orig))
        plt.subplot(142)
        plt.title("kWTA")
        plt.imshow(torch_to_matplotlib(kwta))
        plt.subplot(143)
        plt.imshow(torch_to_matplotlib(restored))
        plt.title("Restored")
        if debug:
            plt.subplot(144)
            plt.imshow(torch_to_matplotlib(raw))
            plt.title("Before inverse")
        plt.show()


def surfplot(dataset="MNIST"):
    loader = get_data_loader(dataset=dataset)
    logdim = torch.arange(8, 14)
    embedding_dimensions = torch.pow(2, logdim)
    sparsities = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75]
    channels = 1 if dataset == "MNIST" else 3
    overlap_running_mean = defaultdict(lambda: defaultdict(lambda: defaultdict(MeanOnline)))
    for images, labels in tqdm(loader, desc=f"kWTA inverse overlap surfplot ({dataset})"):
        if torch.cuda.is_available():
            images = images.cuda()
        for channel in range(channels):
            images_channel = images[:, channel, :, :]
            images_binary = (images_channel > 0).type(torch.float32).flatten(start_dim=1)
            n_bits_active = images_binary.sum(dim=1)
            sparsity_channel = images_binary.mean()
            for i, embedding_dim in enumerate(embedding_dimensions):
                for j, sparsity in enumerate(sparsities):
                    weights = torch.randn(images_binary.shape[1], embedding_dim, device=images_binary.device)
                    embeddings = images_binary @ weights
                    kwta_embeddings_channel = _KWinnersTakeAllFunction.apply(embeddings, sparsity)
                    before_inverse_channel = kwta_embeddings_channel @ weights.transpose(0, 1)
                    restored_channel = _KWinnersTakeAllFunction.apply(before_inverse_channel, sparsity_channel)
                    overlap_batch = (restored_channel == images_binary).sum(dim=1).type(torch.float) / n_bits_active
                    overlap_running_mean[channel][i][j].update(overlap_batch.mean())
    overlap = torch.empty(channels, len(embedding_dimensions), len(sparsities))
    for channel in range(overlap.shape[0]):
        for i in range(overlap.shape[1]):
            for j in range(overlap.shape[2]):
                overlap[channel, i, j] = overlap_running_mean[channel][i][j].get_mean()
    overlap = overlap.mean(dim=0)

    viz = VisdomMighty(env="kWTA inverse")
    opts = dict(
        title=f"kWTA inverse overlap: {dataset}",
        ytickvals=list(range(len(embedding_dimensions))),
        yticklabels=[f'2^{power}' for power in logdim],
        ylabel='embedding_dim',
        xtickvals=list(range(len(sparsities))),
        xticklabels=list(map(str, sparsities)),
        xlabel='sparsity',
    )
    viz.contour(X=overlap, win=f'overlap contour: {dataset}', opts=opts)
    viz.surf(X=overlap, win=f'overlap surf: {dataset}', opts=opts)


if __name__ == '__main__':
    # kwta_inverse()
    surfplot()
