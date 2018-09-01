from pathlib import Path
import math

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models_bin"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

SPARSITY = 0.05
EMBEDDING_SIZE = 128
MAX_L0_DIST = 2 * math.ceil(SPARSITY * EMBEDDING_SIZE)
