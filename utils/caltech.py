import os
import random
import shutil
import tarfile
import warnings
from pathlib import Path

import requests
import torch.utils.data
import torchvision
from torchvision.datasets.utils import check_integrity
from torchvision.datasets.folder import default_loader, IMG_EXTENSIONS
from tqdm import tqdm, trange

from utils.constants import DATA_DIR

CALTECH_RAW = DATA_DIR / "Caltech_raw"
CALTECH_256 = DATA_DIR / "Caltech256"
CALTECH_10 = DATA_DIR / "Caltech10"

CALTECH_URL = "http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar"
TAR_FILEPATH = CALTECH_RAW / CALTECH_URL.split('/')[-1]


def download():
    CALTECH_RAW.mkdir(parents=True, exist_ok=True)
    if check_integrity(TAR_FILEPATH, md5="67b4f42ca05d46448c6bb8ecd2220f6d"):
        print(f"Using downloaded and verified {TAR_FILEPATH}")
    else:
        request = requests.get(CALTECH_URL, stream=True)
        total_size = int(request.headers.get('content-length', 0))
        wrote_bytes = 0
        with open(TAR_FILEPATH, 'wb') as f:
            for data in tqdm(request.iter_content(chunk_size=1024), desc=f"Downloading {CALTECH_URL}",
                             total=total_size // 1024,
                             unit='KB', unit_scale=True):
                wrote_bytes += f.write(data)
        if wrote_bytes != total_size:
            warnings.warn("Content length mismatch. Try downloading again.")
    print(f"Extracting {TAR_FILEPATH}")
    with tarfile.open(TAR_FILEPATH) as tar:
        tar.extractall(path=CALTECH_RAW)


def move_files(filepaths, folder_to: Path):
    folder_to.mkdir(parents=True, exist_ok=True)
    for filepath in filepaths:
        filepath.rename(folder_to / filepath.name)


def split_train_test(train_part=0.8):
    # we don't need noise/background class
    caltech_root = CALTECH_RAW / TAR_FILEPATH.stem
    shutil.rmtree(caltech_root / "257.clutter", ignore_errors=True)
    for category in caltech_root.iterdir():
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


class Caltech256(torchvision.datasets.DatasetFolder):
    def __init__(self, train=True, transformed=True, root=CALTECH_256):
        fold = "train" if train else "test"
        self.root = root / fold
        self.prepare()
        if transformed:
            self.transform_images()
            super().__init__(root=self.root_transformed, loader=torch.load, extensions=['.pt'])
        else:
            super().__init__(root=self.root, loader=default_loader, extensions=IMG_EXTENSIONS,
                             transform=self.transform_caltech)

    def prepare(self):
        if not CALTECH_256.exists():
            download()
            split_train_test()

    @property
    def root_transformed(self):
        return self.root.with_name(self.root.name + "_transformed")

    @property
    def transform_caltech(self):
        return torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def transform_images(self):
        if self.root_transformed.exists():
            return
        dataset = torchvision.datasets.ImageFolder(root=self.root, transform=self.transform_caltech)
        for sample_id in trange(len(dataset.samples), desc=f"Applying image transform {self.root}"):
            image_path, class_id = dataset.samples[sample_id]
            image = dataset.loader(image_path)
            image = dataset.transform(image)
            transformed_path = self.root_transformed / dataset.classes[class_id] / Path(image_path).name
            transformed_path = transformed_path.with_suffix('.pt')
            transformed_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(image, transformed_path)


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
