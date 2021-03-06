import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10

import models.caltech
import models.cifar
from mighty.loss import *
from mighty.models import MLP, Flatten
from mighty.trainer import *
from mighty.utils.common import set_seed
from mighty.utils.data import DataLoader, TransformDefault
from mighty.utils.domain import MonitorLevel
from mighty.monitor.mutual_info import *
from models import *
from monitor.accuracy import AccuracyEmbeddingKWTA
from trainer import *
from utils.caltech import Caltech10


def get_optimizer_scheduler(model: nn.Module):
    optimizer = torch.optim.Adam(
        filter(lambda param: param.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-5
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=15,
                                                           threshold=1e-3,
                                                           min_lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return optimizer, scheduler


def train_grad(n_epoch=30, dataset_cls=MNIST):
    model = MLP(784, 128, 10)
    optimizer, scheduler = get_optimizer_scheduler(model)
    criterion = nn.CrossEntropyLoss()
    data_loader = DataLoader(dataset_cls,
                             transform=TransformDefault.mnist())
    trainer = TrainerGrad(model, criterion=criterion, data_loader=data_loader,
                          optimizer=optimizer, scheduler=scheduler)
    trainer.restore()  # uncomment to restore the saved state
    # trainer.monitor.advanced_monitoring(level=MonitorLevel.SIGNAL_TO_NOISE)
    trainer.train(n_epochs=n_epoch)


def train_autoenc(n_epoch=60, dataset_cls=MNIST):
    kwta = KWinnersTakeAllSoft(hardness=2, sparsity=0.05)
    # kwta = SynapticScaling(kwta, synaptic_scale=3.0)
    model = AutoencoderLinearKWTA(input_dim=784, encoding_dim=256, kwta=kwta)
    # model = MLP(784, 128, 64)
    if isinstance(model, AutoencoderLinearKWTATanh):
        # normalize in range [-1, 1]
        normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
        criterion = nn.MSELoss()
        reconstr_thr = torch.linspace(-0.5, 0.9, steps=10, dtype=torch.float32)
    else:
        normalize = None
        criterion = nn.BCEWithLogitsLoss()
        reconstr_thr = torch.linspace(0.1, 0.95, steps=10, dtype=torch.float32)
    optimizer, scheduler = get_optimizer_scheduler(model)
    data_loader = DataLoader(dataset_cls,
                             transform=TransformDefault.mnist(
                                 normalize=normalize
                             ))
    kwta_scheduler = KWTAScheduler(model=model, step_size=10,
                                   gamma_sparsity=0.7, min_sparsity=0.05,
                                   gamma_hardness=2, max_hardness=10)
    trainer = TrainerAutoencoderBinary(model,
                                       criterion=criterion,
                                       data_loader=data_loader,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       kwta_scheduler=kwta_scheduler,
                                       reconstruct_threshold=reconstr_thr,
                                       accuracy_measure=AccuracyEmbeddingKWTA(
                                           cache=model.encoding_dim <= 2048))
    # trainer.restore()  # uncomment to restore the saved state
    # trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
    trainer.train(n_epochs=n_epoch)


def train_kwta_autoenc(dataset_cls=MNIST):
    kwta = KWinnersTakeAllSoft(sparsity=0.05, hardness=2)
    model = AutoencoderLinearKWTA(784, 2048, kwta)
    data_loader = DataLoader(dataset_cls,
                             transform=transforms.ToTensor(),
                             eval_size=10000)
    criterion = nn.BCEWithLogitsLoss()
    optimizer, scheduler = get_optimizer_scheduler(model)
    kwta_scheduler = KWTAScheduler(model=model, step_size=1,
                                   gamma_sparsity=0.7, min_sparsity=0.05,
                                   gamma_hardness=1.25, max_hardness=20)
    trainer = TrainerAutoencoderBinary(model,
                                       criterion=criterion,
                                       data_loader=data_loader,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       kwta_scheduler=kwta_scheduler,
                                       accuracy_measure=AccuracyEmbeddingKWTA(
                                           cache=True
                                       ))
    # trainer.restore()
    # trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
    # trainer.watch_modules = (KWinnersTakeAll,)
    trainer.train(n_epochs=40)


def test(model, n_epoch=500, dataset_cls=MNIST):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    criterion = nn.CrossEntropyLoss()
    data_loader = DataLoader(dataset_cls,
                             transform=TransformDefault.mnist())
    trainer = Test(model=model, criterion=criterion, data_loader=data_loader)
    trainer.train(n_epochs=n_epoch)


def train_kwta(n_epoch=500, dataset_cls=MNIST):
    kwta = KWinnersTakeAllSoft(sparsity=0.05, hard=False, hardness=2)
    # kwta = SynapticScaling(kwta, synaptic_scale=3)
    model = MLP_kWTA(784, 64, 256, kwta=kwta)
    optimizer, scheduler = get_optimizer_scheduler(model)
    criterion = TripletLossSampler(TripletCosineLoss(margin=0.5))
    data_loader = DataLoader(dataset_cls,
                             transform=TransformDefault.mnist())
    kwta_scheduler = KWTAScheduler(model=model, step_size=10,
                                   gamma_sparsity=0.7, min_sparsity=0.05,
                                   gamma_hardness=2, max_hardness=20)
    trainer = TrainerEmbeddingKWTA(model=model,
                                   criterion=criterion,
                                   data_loader=data_loader,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   kwta_scheduler=kwta_scheduler,
                                   accuracy_measure=AccuracyEmbeddingKWTA(
                                       cache=True),
                                   env_suffix='')
    # trainer.restore()
    trainer.train(n_epochs=n_epoch, mutual_info_layers=1)


def train_pretrained(n_epoch=500, dataset_cls=CIFAR10):
    model = models.cifar.CIFAR10(pretrained=True)
    for param in model.parameters():
        param.requires_grad_(False)
    kwta = KWinnersTakeAllSoft(sparsity=0.3)
    model.classifier = nn.Sequential(nn.Linear(1024, 128, bias=False), kwta)
    optimizer, scheduler = get_optimizer_scheduler(model)
    criterion = ContrastiveLossSampler(nn.CosineEmbeddingLoss(margin=0.5))
    data_loader = DataLoader(dataset_cls,
                             transform=TransformDefault.cifar10())
    kwta_scheduler = KWTAScheduler(model=model, step_size=15,
                                   gamma_sparsity=0.7, min_sparsity=0.05,
                                   gamma_hardness=2, max_hardness=10)
    trainer = TrainerEmbedding(model=model,
                               criterion=criterion,
                               data_loader=data_loader,
                               optimizer=optimizer,
                               scheduler=scheduler,
                               kwta_scheduler=kwta_scheduler,
                               accuracy_measure=AccuracyEmbeddingKWTA(),
                               env_suffix='')
    trainer.train(n_epochs=n_epoch, mutual_info_layers=1)


def train_caltech(n_epoch=500, dataset_cls=Caltech10):
    dataset_name = dataset_cls.__name__
    models.caltech.set_out_features(key='softmax',
                                    value=int(dataset_name.lstrip("Caltech")))
    kwta = KWinnersTakeAllSoft(sparsity=0.05)
    model = models.caltech.resnet18(kwta=kwta)
    data_loader = DataLoader(dataset_cls)
    if kwta:
        criterion = ContrastiveLossSampler(nn.CosineEmbeddingLoss(margin=0.5))
        optimizer, scheduler = get_optimizer_scheduler(model)
        kwta_scheduler = KWTAScheduler(model=model, step_size=10,
                                       gamma_sparsity=0.7, min_sparsity=0.05,
                                       gamma_hardness=2, max_hardness=20)
        trainer = TrainerEmbeddingKWTA(model=model, criterion=criterion,
                                       data_loader=data_loader,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       kwta_scheduler=kwta_scheduler)
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = get_optimizer_scheduler(model)
        trainer = TrainerGrad(model=model, criterion=criterion,
                              data_loader=data_loader, optimizer=optimizer,
                              scheduler=scheduler)
    trainer.train(n_epochs=n_epoch)


if __name__ == '__main__':
    set_seed(26)
    # torch.backends.cudnn.benchmark = True
    train_kwta_autoenc()
