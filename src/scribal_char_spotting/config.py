# DAY 1

# Variables

# config.py
# Holds all paths, tiling parameters, and dataset constants

# HOW TO USE IN OTHER FILES:
# from scribal_char_spotting.config import COCO_PATH, TILE_SIZE, OVERLAP
# or to import everything:
# import scribal_char_spotting.config as cfg  → then use cfg.TILE_SIZE etc.

import os

# ── ROOT ────────────────────────────────────────────────────────────────────
SOURCE_PATH = "C:/Users/addod/scribal-glyph-character-spotting"

# ── INPUT DATA ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(SOURCE_PATH, "data", "training-25plus")
COCO_PATH = os.path.join(DATA_PATH, "coco.json")
PSEUDO_YOLO_PATH = os.path.join(DATA_PATH, "pseudo_YOLO")
IMAGE_PATH = os.path.join(DATA_PATH, "untiled_images")

# ── SPLITS ───────────────────────────────────────────────────────────────────
SPLITS_PATH = os.path.join(SOURCE_PATH, "splits")
TRAIN_SPLIT = os.path.join(SPLITS_PATH, "train.uttlist")
VAL_SPLIT = os.path.join(SPLITS_PATH, "validation.uttlist")
TEST_SPLIT = os.path.join(SPLITS_PATH, "test.uttlist")

# ── OUTPUT ───────────────────────────────────────────────────────────────────
TILE_STORAGE_PATH = os.path.join(SOURCE_PATH, "data", "tiled_images")
TILE_LABEL_PATH = os.path.join(SOURCE_PATH, "data", "tiled_labels")

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
YOLO_PATH = os.path.join(SOURCE_PATH, "YOLO_training")
YOLO_YAML_PATH = os.path.join(SOURCE_PATH, "configs", "scribal-glyph-charspotting.yaml")
YOLO_SAVE_PATH = os.path.join(YOLO_PATH, "saved_models")


# ── TILING PARAMETERS ────────────────────────────────────────────────────────
TILE_SIZE = 512
OVERLAP = 128  # 25% of TILE_SIZE
STRIDE = TILE_SIZE - OVERLAP  # 384

# ── YOLO TRAINING PARAMETERS ─────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.25  # minimum detection confidence during inference
IOU_THRESHOLD = 0.5  # IoU threshold for duplicate removal during un-tiling

# ── SANITY CHECK (runs when you execute config.py directly) ──────────────────
if __name__ == "__main__":
    print("=== Path Check ===")
    paths_to_check = {
        "DATA_PATH": DATA_PATH,
        "COCO_PATH": COCO_PATH,
        "PSEUDO_YOLO_PATH": PSEUDO_YOLO_PATH,
        "IMAGE_PATH": IMAGE_PATH,
        "SPLITS_PATH": SPLITS_PATH,
        "TXTS_PATH": TXTS_PATH,
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
