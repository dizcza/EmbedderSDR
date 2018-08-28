import torch

from model import Embedder
from trainer.gradient import TrainerGrad
from utils import set_seed, ContrastiveLabeledLoss


def train(n_epoch=100):
    set_seed(26)
    model = Embedder()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           threshold=1e-3, min_lr=1e-4)
    trainer = TrainerGrad(model=model, criterion=ContrastiveLabeledLoss(same_only=0), dataset_name="MNIST56",
                          optimizer=optimizer, scheduler=scheduler)
    trainer.train(n_epoch=n_epoch, save=False, epoch_update_step=1, with_mutual_info=1, watch_parameters=True)


if __name__ == '__main__':
    train()
