import os
import torch
import torch.nn as nn

from trainer.kwta import TrainerGradKWTA, KWTAScheduler
from trainer.gradient import TrainerGrad
from utils import set_seed
from loss import ContrastiveLossBatch
from model import *

os.environ['FULL_FORWARD_PASS_SIZE'] = '10000'


def train_grad(n_epoch=500, dataset_name="CIFAR10"):
    model = MobileNet(last_layer=None)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15,
                                                           threshold=1e-3, min_lr=1e-4)
    criterion = ContrastiveLossBatch(metric='cosine', random_pairs=True)
    trainer = TrainerGrad(model=model, criterion=criterion, dataset_name=dataset_name, optimizer=optimizer,
                          scheduler=scheduler)
    trainer.train(n_epoch=n_epoch, epoch_update_step=1, watch_parameters=True, mutual_info_layers=1)


def train_kwta(n_epoch=500, dataset_name="CIFAR10"):
    kwta = KWinnersTakeAllSoft(sparsity=0.3, hardness=1)
    model = EmbedderSDR(last_layer=kwta, dataset_name=dataset_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15,
                                                           threshold=1e-3, min_lr=1e-4)
    criterion = ContrastiveLossBatch(metric='cosine', random_pairs=True)
    kwta_scheduler = KWTAScheduler(model=model, step_size=15, gamma_sparsity=0.5, min_sparsity=0.15,
                                   gamma_hardness=2, max_hardness=10)
    trainer = TrainerGradKWTA(model=model, criterion=criterion, dataset_name=dataset_name, optimizer=optimizer,
                              scheduler=scheduler, kwta_scheduler=kwta_scheduler)
    trainer.train(n_epoch=n_epoch, epoch_update_step=1, watch_parameters=True, mutual_info_layers=1)


if __name__ == '__main__':
    set_seed(26)
    train_kwta()

