import os
from src.definitions import WANDB_DIR

def set_wandb_directories():
    if WANDB_DIR is None:
        return
    os.environ["WANDB_CACHE_DIR"] = os.path.join(WANDB_DIR, 'cache')
    os.environ["WANDB_CONFIG_DIR"] = os.path.join(WANDB_DIR, 'config')