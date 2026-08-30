"""
Paths, tiling parameters and dataset constants.

The repository root is derived from this file's location, so the package works
from any checkout. Override it with the SCRIBAL_ROOT environment variable when
the data lives outside the repo (for example on Colab).

Usage:
    from scribal_char_spotting.config import COCO_PATH, TILE_SIZE, OVERLAP
    import scribal_char_spotting.config as cfg   # then cfg.TILE_SIZE
"""

import os

#  ROOT
#  config.py -> scribal_char_spotting -> src -> repository root
_DEFAULT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
SOURCE_PATH = os.environ.get("SCRIBAL_ROOT", _DEFAULT_ROOT)

#  INPUT DATA
#  Not tracked in git: download the training-25plus set from the FAU competition.
DATA_PATH = os.path.join(SOURCE_PATH, "data", "training-25plus")
COCO_PATH = os.path.join(DATA_PATH, "coco.json")
PSEUDO_YOLO_PATH = os.path.join(DATA_PATH, "pseudo_YOLO")
IMAGE_PATH = os.path.join(DATA_PATH, "untiled_images")

#  OUTPUT
TILE_STORAGE_PATH = os.path.join(SOURCE_PATH, "data", "tiled_images")
TILE_LABEL_PATH = os.path.join(SOURCE_PATH, "data", "tiled_labels")

# DATASET FOLDER PATHS
DATASET_PATH = os.path.join(SOURCE_PATH, "data", "dataset")
DATASET_IMAGES_PATH = os.path.join(DATASET_PATH, "images")
DATASET_LABELS_PATH = os.path.join(DATASET_PATH, "labels")

TRAIN_IMAGES_PATH = os.path.join(DATASET_IMAGES_PATH, "train")
VAL_IMAGES_PATH = os.path.join(DATASET_IMAGES_PATH, "val")
TEST_IMAGES_PATH = os.path.join(DATASET_IMAGES_PATH, "test")

TRAIN_LABELS_PATH = os.path.join(DATASET_LABELS_PATH, "train")
VAL_LABELS_PATH = os.path.join(DATASET_LABELS_PATH, "val")
TEST_LABELS_PATH = os.path.join(DATASET_LABELS_PATH, "test")

TXTS_PATH = os.path.join(SOURCE_PATH, "txts")
LETTER_DICTIONARY_PATH = os.path.join(TXTS_PATH, "letter_dictionary.txt")

# YOLO PATHS
YOLO_PATH = os.path.join(SOURCE_PATH, "YOLO_training")
YOLO_YAML_PATH = os.path.join(SOURCE_PATH, "configs", "scribal-glyph-charspotting.yaml")
YOLO_SAVE_PATH = os.path.join(YOLO_PATH, "saved_models")

#  TILING PARAMETERS
TILE_SIZE = 512
OVERLAP = 128  # 25% of TILE_SIZE
STRIDE = TILE_SIZE - OVERLAP  # 384

#  DE-TILING
#  IoU threshold for removing duplicate detections across tile boundaries.
#  Matches the iou passed to model.predict() in the training notebook.
IOU_THRESHOLD = 0.45

#  Set SCRIBAL_VERBOSE=1 to re-enable the per-tile progress printing that the
#  pipeline modules used to emit unconditionally.
VERBOSE = os.environ.get("SCRIBAL_VERBOSE", "") not in ("", "0", "false", "False")


def log(*args, **kwargs):
    """Print only when SCRIBAL_VERBOSE is set."""
    if VERBOSE:
        print(*args, **kwargs)


#  SANITY CHECK
if __name__ == "__main__":
    print("=== Path Check ===")
    paths_to_check = {
        "SOURCE_PATH": SOURCE_PATH,
        "DATA_PATH": DATA_PATH,
        "COCO_PATH": COCO_PATH,
        "PSEUDO_YOLO_PATH": PSEUDO_YOLO_PATH,
        "IMAGE_PATH": IMAGE_PATH,
        "TILE_STORAGE_PATH": TILE_STORAGE_PATH,
        "TILE_LABEL_PATH": TILE_LABEL_PATH,
        "DATASET_PATH": DATASET_PATH,
        "TXTS_PATH": TXTS_PATH,
        "YOLO_PATH": YOLO_PATH,
        "YOLO_YAML_PATH": YOLO_YAML_PATH,
        "YOLO_SAVE_PATH": YOLO_SAVE_PATH,
    }
    for name, path in paths_to_check.items():
        status = "OK" if os.path.exists(path) else "NOT FOUND"
        print(f"  {name:20s}: {status:10s}  ({path})")

    print("\n=== Tiling Parameters ===")
    print(f"  TILE_SIZE : {TILE_SIZE}")
    print(f"  OVERLAP   : {OVERLAP}")
    print(f"  STRIDE    : {STRIDE}")
