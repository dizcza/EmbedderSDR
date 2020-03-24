from collections import defaultdict

import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as F
from PIL import Image
from mighty.monitor.var_online import MeanOnline
from mighty.monitor.viz import VisdomMighty
from mighty.utils.data import DataLoader, get_normalize_inverse
from torchvision import transforms
from torchvision.datasets import MNIST
from tqdm import tqdm

from models.kwta import KWinnersTakeAllFunction
from utils.algebra import factors_root


def torch_to_matplotlib(image):
    if image.shape[0] == 1:
        return image.squeeze(dim=0)
    else:
        return image.transpose(0, 1).transpose(1, 2)


def undo_normalization(images_normalized, normalize_inverse):
    tensor = torch.stack(list(map(normalize_inverse, images_normalized)))
    tensor.clamp_(0, 1)
    return tensor


def kwta_inverse(embedding_dim=10000, sparsity=0.05, dataset_cls=MNIST):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    loader = DataLoader(dataset_cls, normalize=normalize)
    normalize_inverse = get_normalize_inverse(loader.dataset.transform)
    images, labels = next(iter(loader))
    batch_size, channels, height, width = images.shape
    images_binary = (images > 0).type(torch.float32)
    sparsity_input = images_binary.mean()
    print(f"Sparsity input raw image: {sparsity_input:.3f}")
    images_flatten = images.flatten(start_dim=2)
    weights = torch.randn(images_flatten.shape[2], embedding_dim)
    embeddings = images_flatten @ weights
    kwta_embeddings = KWinnersTakeAllFunction.apply(embeddings.clone(), sparsity)
    before_inverse = kwta_embeddings @ weights.transpose(0, 1)
    restored = KWinnersTakeAllFunction.apply(before_inverse.clone(), sparsity_input)

    kwta_embeddings = kwta_embeddings.view(batch_size, channels, *factors_root(embedding_dim))
    restored = restored.view_as(images)

    images = undo_normalization(images, normalize_inverse)

    viz = VisdomMighty(env="kWTA inverse")
    transform = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                torchvision.transforms.Resize(size=128, interpolation=Image.NEAREST),
                                                torchvision.transforms.ToTensor()])
    transformed_images = []
    for orig, kwta, restored in zip(images, kwta_embeddings, restored):
        transformed_images.append(transform(orig))
        transformed_images.append(transform(kwta))
        transformed_images.append(transform(restored))
    transformed_images = torch.stack(transformed_images, dim=0)
    viz.images(transformed_images, win='images', nrow=3, opts=dict(
        title=f"Original | kWTA(n={embedding_dim}, sparsity={sparsity}) | Restored",
    ))


def calc_overlap(vec1, vec2):
    """
    :param vec1: batch of binary vectors
    :param vec2: batch of binary vectors
    :return: (float) vec1 and vec2 similarity (overlap)
    """
    vec1 = vec1.flatten(start_dim=1)
    vec2 = vec2.flatten(start_dim=1)
    k_active = (vec1.sum(dim=1) + vec2.sum(dim=1)) / 2
    similarity = (vec1 * vec2).sum(dim=1) / k_active
    similarity = similarity.mean()
    return similarity


def kwta_translation_similarity(embedding_dim=10000, sparsity=0.05,
                                translate=(1, 1), dataset_cls=MNIST):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    loader = DataLoader(dataset_cls, normalize=normalize)
    images, labels = next(iter(loader))
    images = (images > 0).type(torch.float32)

    images_translated = []
    for im in images:
        im_pil = F.to_pil_image(im)
        im_translated = F.affine(im_pil, angle=0, translate=translate, scale=1, shear=0)
        im_translated = F.to_tensor(im_translated)
        images_translated.append(im_translated)
    images_translated = torch.stack(images_translated, dim=0)
    assert images_translated.unique(sorted=True).tolist() == [0, 1]

    w, h = images.shape[2:]
    weights = torch.randn(w * h, embedding_dim)

    def apply_kwta(images_input):
        """
        :param images_input: (B, C, W, H) images tensor
        :return: (B, C, k_active) kwta encoded SDR tensor
        """
        images_flatten = images_input.flatten(start_dim=2)
        embeddings = images_flatten @ weights
        kwta_embedding = KWinnersTakeAllFunction.apply(embeddings.clone(), sparsity)
        return kwta_embedding

    kwta_orig = apply_kwta(images)
    kwta_translated = apply_kwta(images_translated)

    print(f"input image ORIG vs TRANSLATED similarity {calc_overlap(images, images_translated):.3f}")
    print(f"random-kWTA ORIG vs TRANSLATED similarity: {calc_overlap(kwta_orig, kwta_translated):.3f}")


def surfplot(dataset_cls=MNIST):
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    loader = DataLoader(dataset_cls, normalize=normalize)
    logdim = torch.arange(8, 14)
    embedding_dimensions = torch.pow(2, logdim)
    sparsities = [0.001, 0.01, 0.05, 0.1, 0.3, 0.5, 0.75]
    overlap_running_mean = defaultdict(lambda: defaultdict(MeanOnline))
    for images, labels in tqdm(loader, desc=f"kWTA inverse overlap surfplot "
                                            f"({dataset_cls.__name__})"):
        if torch.cuda.is_available():
            images = images.cuda()
        images_binary = (images > 0).type(torch.float32).flatten(start_dim=2)
        sparsity_channel = images_binary.mean()
        for i, embedding_dim in enumerate(embedding_dimensions):
            for j, sparsity in enumerate(sparsities):
                weights = torch.randn(images_binary.shape[2], embedding_dim, device=images_binary.device)
                embeddings = images_binary @ weights
                kwta_embeddings = KWinnersTakeAllFunction.apply(embeddings, sparsity)
                before_inverse = kwta_embeddings @ weights.transpose(0, 1)
                restored = KWinnersTakeAllFunction.apply(before_inverse, sparsity_channel)
                overlap = calc_overlap(images_binary, restored)
                overlap_running_mean[i][j].update(overlap)
    overlap = torch.empty(len(embedding_dimensions), len(sparsities))
    for i in range(overlap.shape[0]):
        for j in range(overlap.shape[1]):
            overlap[i, j] = overlap_running_mean[i][j].get_mean()

    viz = VisdomMighty(env="kWTA inverse")
    opts = dict(
        title=f"kWTA inverse overlap: {dataset_cls.__name__}",
        ytickvals=list(range(len(embedding_dimensions))),
        yticklabels=[f'2^{power}' for power in logdim],
        ylabel='embedding_dim',
        xtickvals=list(range(len(sparsities))),
        xticklabels=list(map(str, sparsities)),
        xlabel='sparsity',
    )
    viz.contour(X=overlap, win=f'overlap contour: {dataset_cls.__name__}', opts=opts)
    viz.surf(X=overlap, win=f'overlap surf: {dataset_cls.__name__}', opts=opts)


if __name__ == '__main__':
    kwta_inverse()
    surfplot()
    kwta_translation_similarity()
