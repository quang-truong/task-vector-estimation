import os
from rootutils import find_root

ROOT_DIR = find_root(indicator=".project-root")
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
SRC_DIR = os.path.join(ROOT_DIR, 'src')
WEIGHT_DIR = os.path.join(ROOT_DIR, 'weights')
WANDB_DIR = os.path.join('.cache')    # Change this to your own directory
CFG_DIR = os.path.join(ROOT_DIR, 'cfg')