import torch

from model import Embedder
from trainer.gradient import TrainerGrad
from utils import set_seed
from loss import ContrastiveLossBatch, ContrastiveLossAnchor


def train(n_epoch=100):
    set_seed(26)
    model = Embedder()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           threshold=1e-3, min_lr=1e-5)
    trainer = TrainerGrad(model=model, criterion=ContrastiveLossBatch(same_only=False), dataset_name="FashionMNIST",
                          optimizer=optimizer, scheduler=scheduler)
    trainer.train(n_epoch=n_epoch, epoch_update_step=1, with_mutual_info=True, watch_parameters=True)


if __name__ == '__main__':
    train()
