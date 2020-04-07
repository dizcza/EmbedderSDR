import os

# os.environ['FULL_FORWARD_PASS_SIZE'] = '10000'

import torch
import torch.nn as nn
import torchvision.models
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST, Caltech256

from PIL import Image

import models.caltech
import models.cifar
from mighty.loss import *
from models import *
from mighty.monitor.accuracy import AccuracyArgmax
from monitor.accuracy import AccuracyEmbeddingKWTA, AccuracyAutoencoderBinary, \
    AccuracyEmbeddingLISTA, AccuracyAutoencoder
from mighty.monitor.monitor import Monitor
from mighty.monitor.mutual_info import *
from mighty.trainer import *
from trainer import *
from mighty.utils.common import set_seed
from utils.constants import IMAGES_DIR
from mighty.utils.domain import MonitorLevel
from mighty.utils.data import NormalizeInverse, DataLoader
from mighty.utils.hooks import DumpActivationsHook
from mighty.utils.stub import OptimizerStub


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
    return optimizer, scheduler


def train_mask():
    """
    Train explainable mask for an image from ImageNet, using pretrained model.
    """
    model = torchvision.models.vgg19(pretrained=True)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    normalize = transforms.Normalize(
        # ImageNet normalize
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([transforms.Resize(size=(224, 224)),
                                    transforms.ToTensor(), normalize])
    accuracy_measure = AccuracyArgmax()
    monitor = Monitor(
        accuracy_measure=accuracy_measure,
        mutual_info=MutualInfoKMeans(),
        normalize_inverse=NormalizeInverse(mean=normalize.mean,
                                           std=normalize.std),
    )
    monitor.open(env_name='mask')
    image = Image.open(IMAGES_DIR / "flute.jpg")
    image = transform(image)
    mask_trainer = MaskTrainer(
        accuracy_measure=accuracy_measure,
        image_shape=image.shape,
        show_progress=True
    )
    monitor.log(repr(mask_trainer))
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    outputs = model(image.unsqueeze(dim=0))
    proba = accuracy_measure.predict_proba(outputs)
    proba_max, label_true = proba[0].max(dim=0)
    print(f"True label: {label_true} (confidence {proba_max: .5f})")
    monitor.plot_mask(model=model, mask_trainer=mask_trainer, image=image,
                      label=label_true)


def train_grad(n_epoch=30, dataset_cls=MNIST):
    model = MLP(784, 64, 10)
    optimizer, scheduler = get_optimizer_scheduler(model)
    criterion = nn.CrossEntropyLoss()
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    data_loader = DataLoader(dataset_cls, normalize=normalize)
    trainer = TrainerGrad(model, criterion=criterion, data_loader=data_loader,
                          optimizer=optimizer, scheduler=scheduler)
    trainer.restore()  # uncomment to restore the saved state
    # trainer.monitor.advanced_monitoring(level=MonitorLevel.SIGNAL_TO_NOISE)
    trainer.train(n_epochs=n_epoch, mutual_info_layers=0)


def train_autoenc(n_epoch=60, dataset_cls=MNIST):
    kwta = KWinnersTakeAllSoft(hardness=2, sparsity=0.05)
    # kwta = SynapticScaling(kwta, synaptic_scale=3.0)
    model = AutoEncoderLinear(input_dim=784, encoding_dim=256, kwta=kwta)
    # model = MLP(784, 128, 64)
    if isinstance(model, AutoEncoderLinearTanh):
        # normalize in range [-1, 1]
        normalize = transforms.Normalize(mean=(0.5,), std=(0.5,))
        criterion = nn.MSELoss()
        reconstr_thr = torch.linspace(-0.5, 0.9, steps=10, dtype=torch.float32)
    else:
        normalize = None
        criterion = nn.BCEWithLogitsLoss()
        reconstr_thr = torch.linspace(0.1, 0.95, steps=10, dtype=torch.float32)
    optimizer, scheduler = get_optimizer_scheduler(model)
    data_loader = DataLoader(dataset_cls, normalize=normalize, batch_size=4096)
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
                                       accuracy_measure=AccuracyAutoencoderBinary(
                                           cache=model.encoding_dim <= 2048))
    # trainer.restore()  # uncomment to restore the saved state
    # trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
    trainer.train(n_epochs=n_epoch, mutual_info_layers=0)


def test_matching_pursuit_lambdas(dataset_cls=MNIST):
    # os.environ['FULL_FORWARD_PASS_SIZE'] = '10000'
    model = MatchingPursuit(784, 2048)
    data_loader = DataLoader(dataset_cls, normalize=None, batch_size=4096)
    bmp_lambdas = torch.linspace(0.05, 0.95, steps=10)
    trainer = TestMatchingPursuitParameters(model,
                                            criterion=nn.MSELoss(),
                                            data_loader=data_loader,
                                            bmp_params_range=bmp_lambdas,
                                            param_name='lambda')
    trainer.train(n_epochs=1, mutual_info_layers=0)


def train_lista(dataset_cls=MNIST):
    os.environ['FULL_FORWARD_PASS_SIZE'] = '1000'
    model = LISTA(784, 256)
    data_loader = DataLoader(dataset_cls, normalize=None, batch_size=256)
    optimizer, scheduler = get_optimizer_scheduler(model)
    reconstr_thr = torch.linspace(-0.5, 0.5, steps=3, dtype=torch.float32)
    trainer = TrainerLISTA(model,
                           criterion=nn.MSELoss(),
                           data_loader=data_loader,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           accuracy_measure=AccuracyEmbeddingLISTA(
                               cache=model.out_features <= 2048),
                           reconstruct_threshold=reconstr_thr)
    trainer.train(n_epochs=10, mutual_info_layers=0)


def test_binary_matching_pursuit_sparsity(dataset_cls=MNIST):
    os.environ['FULL_FORWARD_PASS_SIZE'] = '100'
    kwta = KWinnersTakeAll(sparsity=None)  # sparsity is a variable param
    model = BinaryMatchingPursuit(784, 2048, kwta=kwta)
    data_loader = DataLoader(dataset_cls, normalize=None, batch_size=32)
    bmp_sparsity = torch.linspace(0.005, 0.25, steps=2)
    trainer = TestMatchingPursuitParameters(model,
                                            criterion=nn.MSELoss(),
                                            data_loader=data_loader,
                                            bmp_params_range=bmp_sparsity,
                                            param_name='sparsity')
    trainer.train(n_epochs=1, mutual_info_layers=0)


def train_matching_pursuit(dataset_cls=MNIST):
    os.environ['FULL_FORWARD_PASS_SIZE'] = '10000'
    model = MatchingPursuit(784, 256, lamb=0.2)
    data_loader = DataLoader(dataset_cls, normalize=None, batch_size=256)
    criterion = LossPenalty(nn.MSELoss(), lambd=model.lambd)
    optimizer, scheduler = get_optimizer_scheduler(model)
    trainer = TrainerAutoencoderBinary(model,
                                       criterion=criterion,
                                       data_loader=data_loader,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       accuracy_measure=AccuracyAutoencoder(
                                           cache=True
                                       ))
    # trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
    trainer.train(n_epochs=10, mutual_info_layers=0)


def train_kwta_autoenc(dataset_cls=MNIST):
    os.environ['FULL_FORWARD_PASS_SIZE'] = '10000'
    embedding_size = 2048
    kwta = KWinnersTakeAllSoft(threshold_size=embedding_size, hardness=2)
    # kwta = KWinnersTakeAllSoft(sparsity=0.05, hardness=2)
    model = MLP_kWTA_Autoenc(784, embedding_size, kwta)
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    data_loader = DataLoader(dataset_cls, normalize=normalize, batch_size=256)
    criterion = nn.MSELoss()
    if kwta.sparsity is None:
        # threshold is used
        criterion = LossPenalty(criterion, lambd=0.002, latent_grad=True)
    optimizer, scheduler = get_optimizer_scheduler(model)
    kwta_scheduler = KWTAScheduler(model=model, step_size=10,
                                   gamma_sparsity=0.7, min_sparsity=0.05,
                                   gamma_hardness=2, max_hardness=10)
    trainer = TrainerAutoencoderBinary(model,
                                       criterion=criterion,
                                       data_loader=data_loader,
                                       optimizer=optimizer,
                                       scheduler=scheduler,
                                       kwta_scheduler=kwta_scheduler,
                                       accuracy_measure=AccuracyAutoencoderBinary(
                                           cache=True
                                       ))
    # trainer.restore()
    trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
    trainer.watch_modules = (KWinnersTakeAll,)
    trainer.train(n_epochs=40, mutual_info_layers=0)


def test_binary_matching_pursuit(dataset_cls=MNIST):
    os.environ['FULL_FORWARD_PASS_SIZE'] = '100'
    kwta = KWinnersTakeAll(sparsity=0.1)
    model = BinaryMatchingPursuit(784, 2048, kwta=kwta)
    data_loader = DataLoader(dataset_cls, normalize=None, batch_size=32)
    trainer = TestMatchingPursuit(model,
                                  criterion=nn.MSELoss(),
                                  data_loader=data_loader,
                                  optimizer=OptimizerStub())
    trainer.train(n_epochs=1, mutual_info_layers=0)


def test(model, n_epoch=500, dataset_cls=MNIST):
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    criterion = nn.CrossEntropyLoss()
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    data_loader = DataLoader(dataset_cls, normalize=normalize)
    trainer = Test(model=model, criterion=criterion, data_loader=data_loader)
    trainer.train(n_epochs=n_epoch, adversarial=True, mask_explain=True)


def train_kwta(n_epoch=500, dataset_cls=MNIST):
    os.environ['FULL_FORWARD_PASS_SIZE'] = '1000'
    kwta = KWinnersTakeAllSoft(sparsity=0.05)
    # kwta = SynapticScaling(kwta, synaptic_scale=3)
    model = MLP_kWTA(784, 256, kwta)
    optimizer, scheduler = get_optimizer_scheduler(model)
    criterion = TripletLoss(metric='cosine')
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    data_loader = DataLoader(dataset_cls, normalize=normalize, batch_size=10)
    kwta_scheduler = KWTAScheduler(model=model, step_size=15,
                                   gamma_sparsity=0.7, min_sparsity=0.05,
                                   gamma_hardness=2, max_hardness=10)
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
    trainer.monitor.advanced_monitoring(level=MonitorLevel.FULL)
    trainer.train(n_epochs=n_epoch, mutual_info_layers=0)


def train_pretrained(n_epoch=500, dataset_cls=CIFAR10):
    model = models.cifar.CIFAR10(pretrained=True)
    for param in model.parameters():
        param.requires_grad_(False)
    kwta = KWinnersTakeAllSoft(sparsity=0.3)
    model.classifier = nn.Sequential(nn.Linear(1024, 128, bias=False), kwta)
    optimizer, scheduler = get_optimizer_scheduler(model)
    criterion = ContrastiveLossRandom(metric='cosine')
    normalize = transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                     std=(0.247, 0.243, 0.261))
    data_loader = DataLoader(dataset_cls, normalize=normalize)
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
    trainer.train(n_epochs=n_epoch, mutual_info_layers=1, mask_explain=False)


def train_caltech(n_epoch=500, dataset_cls=Caltech256):
    dataset_name = dataset_cls.__name__
    models.caltech.set_out_features(key='softmax',
                                    value=int(dataset_name.lstrip("Caltech")))
    kwta = None
    kwta = KWinnersTakeAllSoft(sparsity=0.3)
    model = models.caltech.resnet18(kwta=kwta)
    data_loader = DataLoader(dataset_cls, normalize=None)
    if kwta:
        criterion = ContrastiveLossRandom(metric='cosine')
        optimizer, scheduler = get_optimizer_scheduler(model)
        kwta_scheduler = KWTAScheduler(model=model, step_size=15,
                                       gamma_sparsity=0.7, min_sparsity=0.05,
                                       gamma_hardness=2, max_hardness=10)
        trainer = TrainerEmbedding(model=model, criterion=criterion,
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
    trainer.train(n_epochs=n_epoch, mutual_info_layers=0, mask_explain=False)


def dump_activations(n_epoch=2, dataset_cls=MNIST):
    model = MLP(784, 128, 32, 10)
    optimizer, scheduler = get_optimizer_scheduler(model)
    criterion = nn.CrossEntropyLoss()
    normalize = transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    data_loader = DataLoader(dataset_cls, normalize=normalize)
    trainer = TrainerGrad(model=model, criterion=criterion,
                          data_loader=data_loader, optimizer=optimizer,
                          scheduler=scheduler)
    trainer.train(n_epochs=n_epoch, mutual_info_layers=0)

    # register forward hook
    dumper = DumpActivationsHook(model)

    # trigger hooks
    trainer.run_idle(n_epoch=1)

    # remove hooks when finished and continue training, if needed
    dumper.remove_hooks()


if __name__ == '__main__':
    set_seed(26)
    # torch.backends.cudnn.benchmark = True
    # train_lista()
    # test_binary_matching_pursuit()
    # train_matching_pursuit()
    train_kwta_autoenc()
    # train_autoenc()
    # dump_activations()
    # train_grad()
    # test()
    # train_pretrained()
    # train_caltech()
