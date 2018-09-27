import torch
import torch.nn as nn
import torch.nn.functional

from monitor.accuracy import Accuracy


def tv_norm(mask_expanded, tv_beta: int):
    mask = mask_expanded[0, 0, ::]
    row_grad = (mask[:-1, :] - mask[1:, :]).abs().pow(tv_beta).mean()
    col_grad = (mask[:, :-1] - mask[:, 1:]).abs().pow(tv_beta).mean()
    return row_grad + col_grad


def create_gaussian_filter(size: int, sigma: float, channels: int):
    linspace = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    # Create a x, y coordinate grid of shape (size, size, 2)
    x_grid = linspace.repeat(size).view(size, size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    gaussian_kernel = torch.exp(-xy_grid.pow(2).sum(dim=-1) / (2 * sigma ** 2))
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel /= gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.expand(channels, 1, *gaussian_kernel.shape)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=size, groups=channels, bias=False)
    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad_(False)
    return gaussian_filter


class MaskTrainer:
    """
    Interpretable Explanations of Black Boxes by Meaningful Perturbation.
    https://arxiv.org/pdf/1704.03296.pdf
    """

    tv_beta = 3
    learning_rate = 0.1
    max_iterations = 20
    l1_coeff = 0.01
    tv_coeff = 0.2
    mask_size = 10

    def __init__(self, accuracy_measure: Accuracy, channels: int):
        self.gaussian_filter = create_gaussian_filter(size=5, sigma=3, channels=channels)
        self.padding = nn.modules.ReflectionPad2d(padding=(self.gaussian_filter.kernel_size[0] - 1) // 2)
        self.accuracy_measure = accuracy_measure
        if torch.cuda.is_available():
            self.gaussian_filter.cuda()
            self.padding.cuda()

    def train_mask(self, model: nn.Module, image, label_true):
        channels, height, width = image.shape
        image = image.unsqueeze(dim=0)
        image_blurred = self.gaussian_filter(self.padding(image))
        mask = nn.Parameter(torch.ones(self.mask_size, self.mask_size, dtype=torch.float32, device=image.device))
        optimizer = torch.optim.Adam([mask], lr=self.learning_rate)
        loss_trace = []
        mask_upsampled = None
        image_perturbed = None
        for i in range(self.max_iterations):
            mask_upsampled = mask.expand(1, channels, *mask.shape)
            mask_upsampled = nn.functional.interpolate(mask_upsampled, size=(height, width), mode='bilinear',
                                                       align_corners=True)
            optimizer.zero_grad()
            noise = torch.randn_like(image) * 0.2
            image_perturbed = mask_upsampled * image + (1 - mask_upsampled) * image_blurred
            outputs = model(image_perturbed + noise)
            proba = self.accuracy_measure.predict_proba(outputs)[0, label_true]
            loss = self.l1_coeff * (1 - mask_upsampled).abs().mean() + \
                self.tv_coeff * tv_norm(mask_upsampled, self.tv_beta) + proba
            loss.backward()
            optimizer.step()
            mask_upsampled.data.clamp_(0, 1)
            loss_trace.append(loss.item())
        mask_upsampled = mask_upsampled[0].detach().cpu()
        image_perturbed = image_perturbed[0].detach().cpu()
        return mask_upsampled, loss_trace, image_perturbed
