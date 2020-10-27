import random
import shutil
from pathlib import Path

import torchvision

from mighty.utils.constants import DATA_DIR


def copy_files(filepaths, folder_to):
    Path(folder_to).mkdir(exist_ok=True)
    for filepath in filepaths:
        shutil.copyfile(src=filepath, dst=folder_to / filepath.name)


def split_train_test(caltech256_dir, train_part=0.8):
    if not caltech256_dir.exists():
        raise RuntimeError(f"Download http://www.vision.caltech.edu/"
                           f"Image_Datasets/Caltech256/"
                           f"256_ObjectCategories.tar and extract to "
                           f"{caltech256_dir}")
    caltech256_dir = Path(caltech256_dir)
    caltech_train = caltech256_dir.with_name(f"{caltech256_dir.name}-train")
    caltech_test = caltech256_dir.with_name(f"{caltech256_dir.name}-test")
    if caltech_train.exists() and caltech_test.exists():
        return
    print("Splitting Caltech256 dataset into train and test...")
    shutil.rmtree(caltech_train, ignore_errors=True)
    shutil.rmtree(caltech_test, ignore_errors=True)
    caltech_train.mkdir()
    caltech_test.mkdir()
    for category in Path(caltech256_dir).iterdir():
        if category.name == "257.clutter":
            # skip clutter class
            continue
        images = list(filter(lambda filepath: filepath.suffix == '.jpg',
                             category.iterdir()))
        random.shuffle(images)
        n_train = int(train_part * len(images))
        images_train = images[:n_train]
        images_test = images[n_train:]
        copy_files(images_train, caltech_train / category.name)
        copy_files(images_test, caltech_test / category.name)

    caltech10_train = DATA_DIR / f"{Caltech10.name}-train"
    caltech10_test = DATA_DIR / f"{Caltech10.name}-test"

    for category in Caltech10.classes:
        shutil.copytree(src=caltech_train / category,
                        dst=caltech10_train / category)
        shutil.copytree(src=caltech_test / category,
                        dst=caltech10_test / category)
    print("Caltech256 dataset is split.")


class Caltech256(torchvision.datasets.ImageFolder):
    name = "caltech256"

    def __init__(self, root=DATA_DIR, train=True, **kwargs):
        caltech256_dir = Path(root) / "caltech256"
        split_train_test(caltech256_dir)
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
        super().__init__(root=f"{root / self.name}-{fold}",
                         transform=torchvision.transforms.Compose(transforms))


class Caltech10(Caltech256):
    name = "caltech10"

    classes = (
        "169.radio-telescope", "009.bear", "010.beer-mug", "024.butterfly",
        "025.cactus", "028.camel", "030.canoe", "055.dice", "056.dog",
        "060.duck"
    )
