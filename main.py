import torch

from trainer.kwta import TrainerGradKWTA, KWTAScheduler
from utils import set_seed
from loss import ContrastiveLossBatch

from model.dpn import DPN26
from model.embedder import EmbedderSDR
from model.kwta import KWinnersTakeAll, KWinnersTakeAllSoft


def train(n_epoch=500, dataset_name="CIFAR10_56"):
    set_seed(26)
    kwta = KWinnersTakeAllSoft(sparsity=0.3, hardness=0.1)
    model = EmbedderSDR(kwta_layer=kwta, dataset_name=dataset_name)
    # model = DPN26(kwta_layer=kwta)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, weight_decay=1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                           threshold=1e-3, min_lr=1e-4)
    criterion = ContrastiveLossBatch(same_only=False, metric='cosine')
    # criterion = torch.nn.CosineEmbeddingLoss(margin=0.5)
    kwta_scheduler = KWTAScheduler(model=model, step_size=20, gamma_sparsity=0.5, min_sparsity=0.05,
                                   gamma_hardness=2, max_hardness=10)
    trainer = TrainerGradKWTA(model=model, criterion=criterion, dataset_name=dataset_name,
                              optimizer=optimizer, scheduler=scheduler, kwta_scheduler=kwta_scheduler)
    trainer.train(n_epoch=n_epoch, epoch_update_step=1, with_mutual_info=0, watch_parameters=True)


if __name__ == '__main__':
    train()
