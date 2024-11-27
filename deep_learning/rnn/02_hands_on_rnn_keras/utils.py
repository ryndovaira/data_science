import logging
import os

from config import Config

# Get the existing logger configured in main.py
logger = logging.getLogger()


def get_artifacts_dir(base_dir: str, *sub_dirs: str) -> str:
    """Returns the directory path for saving artifacts, ensuring it exists."""
    dir_path = os.path.join(os.getcwd(), Config.ARTIFACTS_DIR, base_dir, *sub_dirs)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def get_artifacts_arch_dir(base_dir: str, *sub_dirs: str) -> str:
    """Returns the directory path for saving artifacts, ensuring it exists."""
    dir_path = os.path.join(
        os.getcwd(), Config.ARTIFACTS_DIR, base_dir, Config.ARCHITECTURE, *sub_dirs
    )
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def checkpoint_path():
    """Returns the file path for saving model checkpoints."""
    save_dir = get_artifacts_arch_dir(Config.CHECKPOINT_DIR)

    return os.path.join(
        save_dir,
        f"model_checkpoint_{Config.TIMESTAMP}_epoch-{{epoch:02d}}_val-loss-{{val_loss:.4f}}.keras",
    )
