from pathlib import Path
import os

ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models_bin"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"
IMAGES_DIR = ROOT_DIR / "images"
DUMPS_DIR = DATA_DIR / "dumps"
PCA_DIR = DATA_DIR / "pca"

SPARSITY = 0.05

BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 256))
