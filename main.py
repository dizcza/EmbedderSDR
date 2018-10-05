import os

import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms
from PIL import Image

from loss import ContrastiveLossBatch
from model import *
from monitor.accuracy import AccuracyArgmax
from monitor.monitor import Monitor
from trainer import *
from utils.common import set_seed
from utils.constants import IMAGES_DIR
from utils.normalize import NormalizeInverse

os.environ['FULL_FORWARD_PASS_SIZE'] = '10000'


def train_mask():
    """
    Train explainable mask for an image from ImageNet, using pretrained model.
    """
    model = torchvision.models.vgg19(pretrained=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(224, 224)),
                                                torchvision.transforms.ToTensor(), normalize])
    accuracy_measure = AccuracyArgmax()
    monitor = Monitor(test_loader=None, accuracy_measure=accuracy_measure, env_name='mask')
    monitor.normalize_inverse = NormalizeInverse(mean=normalize.mean, std=normalize.std)
    image = Image.open(IMAGES_DIR / "flute.jpg")
    image = transform(image)
    mask_trainer = MaskTrainer(accuracy_measure=accuracy_measure, image_shape=image.shape, show_progress=True)
    monitor.log(repr(mask_trainer))
    if torch.cuda.is_available():
        model = model.cuda()
        image = image.cuda()
    outputs = model(image.unsqueeze(dim=0))
    proba = accuracy_measure.predict_proba(outputs)
    proba_max, label_true = proba[0].max(dim=0)
    print(f"True label: {label_true} (confidence {proba_max: .5f})")
    monitor.plot_mask(model=model, mask_trainer=mask_trainer, image=image, label=label_true)


def train_grad(n_epoch=500, dataset_name="CIFAR10_56"):
    model = EmbedderSDR(last_layer=nn.Linear(128, 2), dataset_name=dataset_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15,
                                                           threshold=1e-3, min_lr=1e-4)
    # criterion = ContrastiveLossBatch(metric='cosine')
    criterion = nn.CrossEntropyLoss()
    trainer = TrainerGrad(model=model, criterion=criterion, dataset_name=dataset_name, optimizer=optimizer,
                          scheduler=scheduler)
    trainer.train(n_epoch=n_epoch, epoch_update_step=1, watch_parameters=True, mutual_info_layers=1)


def train_kwta(n_epoch=500, dataset_name="CIFAR10_56"):
    kwta = KWinnersTakeAllSoft(sparsity=0.3, connect_lateral=False)
    # kwta = SynapticScaling(kwta, synaptic_scale=3)
    model = EmbedderSDR(last_layer=kwta, dataset_name=dataset_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15,
                                                           threshold=1e-3, min_lr=1e-4)
    criterion = ContrastiveLossBatch(metric='cosine')
    kwta_scheduler = KWTAScheduler(model=model, step_size=15, gamma_sparsity=0.5, min_sparsity=0.05,
                                   gamma_hardness=2, max_hardness=10)
    trainer = TrainerGradKWTA(model=model, criterion=criterion, dataset_name=dataset_name, optimizer=optimizer,
                              scheduler=scheduler, kwta_scheduler=kwta_scheduler, env_suffix='')
    trainer.train(n_epoch=n_epoch, epoch_update_step=1, watch_parameters=True, mutual_info_layers=1, mask_explain=True)


def test(n_epoch=500, dataset_name="CIFAR10_56"):
    kwta = KWinnersTakeAllSoft(sparsity=0.3, connect_lateral=False)
    model = EmbedderSDR(last_layer=kwta, dataset_name=dataset_name)
    for param in model.parameters():
        param.requires_grad_(False)
    criterion = ContrastiveLossBatch(metric='cosine')
    trainer = Test(model=model, criterion=criterion, dataset_name=dataset_name)
    trainer.train(n_epoch=n_epoch, epoch_update_step=1, watch_parameters=True, mutual_info_layers=1, adversarial=True,
                  mask_explain=True)


if __name__ == '__main__':
    set_seed(26)
    torch.backends.cudnn.benchmark = True
    # train_kwta()
    # train_mask()
    # train_grad()
    test()
