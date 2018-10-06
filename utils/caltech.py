import os
import random
import shutil
import tarfile
import warnings

import requests
import torchvision
from torchvision.datasets.utils import check_integrity
from tqdm import tqdm

from utils.constants import DATA_DIR

CALTECH_RAW = DATA_DIR / "Caltech_raw"
CALTECH_256 = DATA_DIR / "Caltech256"
CALTECH_10 = DATA_DIR / "Caltech10"


def download():
    url = "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"
    os.makedirs(CALTECH_RAW, exist_ok=True)
    filepath = CALTECH_RAW / url.split('/')[-1]
    if check_integrity(filepath, md5="67b4f42ca05d46448c6bb8ecd2220f6d"):
        print(f"Using downloaded and verified {filepath}")
    else:
        request = requests.get(url, stream=True)
        total_size = int(request.headers.get('content-length', 0))
        wrote_bytes = 0
        with open(filepath, 'wb') as f:
            for data in tqdm(request.iter_content(chunk_size=1024), desc=f"Downloading {url}",
                             total=total_size // 1024,
                             unit='KB', unit_scale=True):
                wrote_bytes += f.write(data)
        if wrote_bytes != total_size:
            warnings.warn("Content length mismatch. Try downloading again.")
    print(f"Extracting {filepath}")
    with tarfile.open(filepath) as tar:
        tar.extractall(path=CALTECH_RAW)


def move_files(filepaths, folder_to):
    os.makedirs(folder_to, exist_ok=True)
    for filepath in filepaths:
        filepath.rename(folder_to / filepath.name)


def split_train_test(train_part=0.8):
    # we don't need noise/background class
    shutil.rmtree(CALTECH_RAW / "257.clutter", ignore_errors=True)
    for category in CALTECH_RAW.iterdir():
        images = list(filter(lambda filepath: filepath.suffix == '.jpg', category.iterdir()))
        random.shuffle(images)
        n_train = int(train_part * len(images))
        images_train = images[:n_train]
        images_test = images[n_train:]
        move_files(images_train, CALTECH_256 / "train" / category.name)
        move_files(images_test, CALTECH_256 / "test" / category.name)
    print("Split Caltech dataset.")


def prepare_subset():
    """
    Prepares Caltech10 data subset.
    """
    subcategories = sorted(os.listdir(CALTECH_256 / "train"))[:10]
    for category in subcategories:
        for fold in ("train", "test"):
            shutil.copytree(CALTECH_256 / fold / category, CALTECH_10 / fold / category)


class Caltech256(torchvision.datasets.ImageFolder):
    def __init__(self, train=True, root=CALTECH_256):
        self.prepare()
        transforms = []
        if train:
            fold = "train"
            transforms.append(torchvision.transforms.RandomHorizontalFlip())
        else:
            fold = "test"
        transforms.extend([
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        super().__init__(root=root / fold, transform=torchvision.transforms.Compose(transforms))

    def prepare(self):
        if not CALTECH_256.exists():
            download()
            split_train_test()


class Caltech10(Caltech256):
    """
    Caltech256 first 10 classes subset.
    """

    def __init__(self, train=True):
        super().__init__(train=train, root=CALTECH_10)

    def prepare(self):
        super().prepare()
        if not CALTECH_10.exists():
            prepare_subset()
