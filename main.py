import os

os.environ['FULL_FORWARD_PASS_SIZE'] = '10000'

import torch
import torch.nn as nn
import torchvision.models
import torchvision.transforms
from PIL import Image

import models.caltech
import models.cifar
from loss import *
from models import *
from monitor.accuracy import AccuracyArgmax
from monitor.monitor import Monitor
from monitor.mutual_info import *
from trainer import *
from utils.common import set_seed
from utils.constants import IMAGES_DIR
from utils.domain import MonitorLevel
from utils.normalize import NormalizeInverse
from utils.hooks import DumpActivationsHook


def get_optimizer_scheduler(model: nn.Module):
    optimizer = torch.optim.Adam(filter(lambda param: param.requires_grad, model.parameters()), lr=1e-3,
                                 weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15,
                                                           threshold=1e-3, min_lr=1e-4)
    return optimizer, scheduler


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
    monitor = Monitor(test_loader=None, accuracy_measure=accuracy_measure, mutual_info=MutualInfoKMeans())
    monitor.open(env_name='mask')
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


def train_grad(n_epoch=500, dataset_name="MNIST"):
    model = EmbedderSDR(last_layer=nn.Sequential(nn.ReLU(inplace=True), nn.Linear(128, 10)), dataset_name=dataset_name)
    optimizer, scheduler = get_optimizer_scheduler(model)
    criterion = nn.CrossEntropyLoss()
    trainer = TrainerGrad(model=model, criterion=criterion, dataset_name=dataset_name, optimizer=optimizer,
                          scheduler=scheduler)
    # trainer.restore()
    trainer.monitor.advanced_monitoring(level=MonitorLevel.SIGNAL_TO_NOISE)
    trainer.train(n_epoch=n_epoch, mutual_info_layers=2)


def train_kwta(n_epoch=500, dataset_name="CIFAR10"):
    kwta = KWinnersTakeAllSoft(sparsity=0.3)
    # kwta = SynapticScaling(kwta, synaptic_scale=3)
    model = EmbedderSDR(last_layer=kwta, dataset_name=dataset_name)
    optimizer, scheduler = get_optimizer_scheduler(model)
    criterion = TripletLoss(metric='cosine')
    kwta_scheduler = KWTAScheduler(model=model, step_size=15, gamma_sparsity=0.3, min_sparsity=0.05,
                                   gamma_hardness=2, max_hardness=10)
    trainer = TrainerGradKWTA(model=model, criterion=criterion, dataset_name=dataset_name, optimizer=optimizer,
                              scheduler=scheduler, kwta_scheduler=kwta_scheduler, env_suffix='')
    # trainer.restore()
    # trainer.monitor.advanced_monitoring(level=MonitorLevel.SIGNAL_TO_NOISE)
    trainer.train(n_epoch=n_epoch, mutual_info_layers=0)


def test(n_epoch=500, dataset_name="CIFAR10"):
    model = models.cifar.CIFAR10(pretrained=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    criterion = nn.CrossEntropyLoss()
    trainer = Test(model=model, criterion=criterion, dataset_name=dataset_name)
    trainer.train(n_epoch=n_epoch, mutual_info_layers=1, adversarial=True, mask_explain=True)


def train_pretrained(n_epoch=500, dataset_name="CIFAR10"):
    model = models.cifar.CIFAR10(pretrained=True)
    for param in model.parameters():
        param.requires_grad_(False)
    kwta = KWinnersTakeAllSoft(sparsity=0.3)
    model.classifier = nn.Sequential(nn.Linear(1024, 128, bias=False), kwta)
    optimizer, scheduler = get_optimizer_scheduler(model)
    criterion = ContrastiveLossRandom(metric='cosine')
    kwta_scheduler = KWTAScheduler(model=model, step_size=15, gamma_sparsity=0.3, min_sparsity=0.05,
                                   gamma_hardness=2, max_hardness=10)
    trainer = TrainerGradKWTA(model=model, criterion=criterion, dataset_name=dataset_name, optimizer=optimizer,
                              scheduler=scheduler, kwta_scheduler=kwta_scheduler)
    trainer.train(n_epoch=n_epoch, mutual_info_layers=1, mask_explain=False)


def train_caltech(n_epoch=500, dataset_name="Caltech256"):
    models.caltech.set_out_features(key='softmax', value=int(dataset_name.lstrip("Caltech")))
    kwta = None
    kwta = KWinnersTakeAllSoft(sparsity=0.3)
    model = models.caltech.resnet18(kwta=kwta)
    if kwta:
        criterion = ContrastiveLossRandom(metric='cosine')
        optimizer, scheduler = get_optimizer_scheduler(model)
        kwta_scheduler = KWTAScheduler(model=model, step_size=15, gamma_sparsity=0.3, min_sparsity=0.05,
                                       gamma_hardness=2, max_hardness=10)
        trainer = TrainerGradKWTA(model=model, criterion=criterion, dataset_name=dataset_name, optimizer=optimizer,
                                  scheduler=scheduler, kwta_scheduler=kwta_scheduler)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = get_optimizer_scheduler(model)
        trainer = TrainerGrad(model=model, criterion=criterion, dataset_name=dataset_name, optimizer=optimizer,
                              scheduler=scheduler)
    trainer.train(n_epoch=n_epoch, mutual_info_layers=0, mask_explain=False)


def dump_activations(n_epoch=2, dataset_name="MNIST"):
    model = MLP(784, 128, 32, 10)
    optimizer, scheduler = get_optimizer_scheduler(model)
    criterion = nn.CrossEntropyLoss()
    trainer = TrainerGrad(model=model, criterion=criterion,
                          dataset_name=dataset_name, optimizer=optimizer,
                          scheduler=scheduler)
    trainer.train(n_epoch=n_epoch, mutual_info_layers=0)

    # register forward hook
    dumper = DumpActivationsHook(model)

    # trigger hooks
    trainer.run_idle(n_epoch=1)

    # remove hooks when finished and continue training, if needed
    dumper.remove_hooks()


if __name__ == '__main__':
    set_seed(26)
    # torch.backends.cudnn.benchmark = True
    # train_kwta()
    dump_activations()
    # test()
    # train_pretrained()
    # train_caltech()
