# scripts/augment_stratified.py
"""
Run the task 3/4 blanking stage against data/dataset_stratified.

run_augmentations reads its source and destination from config at import time,
and those constants point at data/dataset. Repointing config before the import
redirects the stage without editing the tested module.

Writes  data/dataset_stratified/task3/images  387 tiles, all but the labelled boxes blanked
        data/dataset_stratified/task4/images  387 tiles, the labelled boxes blanked
"""
import importlib.util, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import scribal_char_spotting.config as cfg

DEST = os.path.join(ROOT, "data", "dataset_stratified")
cfg.DATASET_PATH = DEST
cfg.DATASET_IMAGES_PATH = os.path.join(DEST, "images")
cfg.DATASET_LABELS_PATH = os.path.join(DEST, "labels")
cfg.TRAIN_IMAGES_PATH = os.path.join(cfg.DATASET_IMAGES_PATH, "train")
cfg.TRAIN_LABELS_PATH = os.path.join(cfg.DATASET_LABELS_PATH, "train")

spec = importlib.util.spec_from_file_location(
    "run_augmentations", os.path.join(ROOT, "scripts", "run_augmentations.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.main()