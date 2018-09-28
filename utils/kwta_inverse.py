import matplotlib.pyplot as plt
import torch
import torch.utils.data

from model.kwta import _KWinnersTakeAllFunction
from utils.normalize import get_normalize_inverse
from utils.common import get_data_loader, factors_root


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
    loader = get_data_loader(dataset=dataset, train=False, batch_size=32)
    normalize_inverse = get_normalize_inverse(loader.dataset.transform)
    images, labels = next(iter(loader))
    batch_size, channels, height, width = images.shape
    k_active = int(embedding_dim * sparsity)
    kwta_embeddings = []
    before_inverse = []
    restored = []
    for channel in range(channels):
        images_channel = images[:, channel, :, :]
        images_binary = (images_channel > 0).type(torch.float32)
        k_active_input = int(images_binary.sum(dim=(1, 2)).mean())
        print(f"Sparsity image raw channel={channel}: {k_active_input / (height * width):.3f}")
        images_flatten = images_channel.view(batch_size, -1)
        weights = torch.randn(images_flatten.shape[1], embedding_dim)
        embeddings = images_flatten @ weights
        kwta_embeddings_channel = _KWinnersTakeAllFunction.apply(embeddings.clone(), k_active)
        before_inverse_channel = kwta_embeddings_channel @ weights.transpose(0, 1)
        restored_channel = _KWinnersTakeAllFunction.apply(before_inverse_channel.clone(), k_active_input)
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


if __name__ == '__main__':
    kwta_inverse()
