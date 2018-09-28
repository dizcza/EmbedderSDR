import os
import torch
import torch.nn as nn

from trainer.kwta import TrainerGradKWTA, KWTAScheduler
from trainer.gradient import TrainerGrad
from utils.common import set_seed
from utils.layers import replace_relu
from loss import ContrastiveLossBatch
from model import *

os.environ['FULL_FORWARD_PASS_SIZE'] = '10000'


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
    model = replace_relu(model, new_relu=kwta)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15,
                                                           threshold=1e-3, min_lr=1e-4)
    criterion = ContrastiveLossBatch(metric='cosine')
    kwta_scheduler = KWTAScheduler(model=model, step_size=15, gamma_sparsity=0.5, min_sparsity=0.05,
                                   gamma_hardness=2, max_hardness=10)
    trainer = TrainerGradKWTA(model=model, criterion=criterion, dataset_name=dataset_name, optimizer=optimizer,
                              scheduler=scheduler, kwta_scheduler=kwta_scheduler, env_suffix='')
    trainer.train(n_epoch=n_epoch, epoch_update_step=1, watch_parameters=True, mutual_info_layers=1)


if __name__ == '__main__':
    set_seed(26)
    train_kwta()

