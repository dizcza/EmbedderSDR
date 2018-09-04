import torch

from model import EmbedderSDR
from trainer.gradient import TrainerGrad
from trainer.kwta import TrainerGradKWTA, KWTAScheduler
from utils import set_seed
from loss import ContrastiveLossBatch, ContrastiveLossAnchor


def train(n_epoch=500, dataset_name="CIFAR10_56"):
    set_seed(26)
    model = EmbedderSDR(dataset_name=dataset_name, sparsity=0.2)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           threshold=1e-3, min_lr=1e-5)
    criterion = ContrastiveLossBatch(same_only=False, metric='cosine')
    # criterion = torch.nn.CosineEmbeddingLoss(margin=0.5)
    kwta_scheduler = KWTAScheduler(model=model, step_size=10, gamma=0.5, min_sparsity=0.05)
    trainer = TrainerGradKWTA(model=model, criterion=criterion, dataset_name=dataset_name,
                              optimizer=optimizer, scheduler=scheduler, kwta_scheduler=kwta_scheduler)
    trainer.train(n_epoch=n_epoch, epoch_update_step=1, with_mutual_info=1, watch_parameters=True)


if __name__ == '__main__':
    train()
