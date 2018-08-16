from pathlib import Path

DATA_DIR = Path(__file__).with_name("data")
MODELS_DIR = DATA_DIR.with_name("models_bin")
CHECKPOINTS_DIR = MODELS_DIR.with_name("checkpoints")
