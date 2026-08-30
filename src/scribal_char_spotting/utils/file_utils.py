"""Helpers for pruning empty tiles and writing the YOLO split manifests."""

import os

import scribal_char_spotting.config as cfg
from scribal_char_spotting.config import log


def generate_split_txts(end_path):
    """
    Write train/val/test manifests listing each split's image paths.

    Paths are sorted and the file ends with a newline, so line counts match the
    number of images and the on-disk order is reproducible. `results_detiler`
    relies on this ordering to match tiles to prediction files.

    Args:
        end_path (str): directory to write the .txt manifests into.
    """
    os.makedirs(end_path, exist_ok=True)

    splits = {
        "train": cfg.TRAIN_IMAGES_PATH,
        "val": cfg.VAL_IMAGES_PATH,
        "test": cfg.TEST_IMAGES_PATH,
    }

    for split_name, split_path in splits.items():
        jpg_paths = sorted(
            f"./images/{split_name}/{f}"
            for f in os.listdir(split_path)
            if f.endswith(".jpg")
        )
        txt_file = os.path.join(end_path, f"{split_name}.txt")
        with open(txt_file, "w", encoding="utf-8", newline="\n") as f:
            for path in jpg_paths:
                f.write(f"{path}\n")

        print(f"Wrote {len(jpg_paths)} paths to {txt_file}")


def remove_empty_tiles(tile_label_path, tile_image_path):
    """
    Delete tiles whose label file holds no annotations, and their images.

    A tile is removed only when both its label and image can be dealt with, so
    the two directories cannot drift out of step.

    Returns:
        int: number of tiles removed.
    """
    number_removed = 0

    for filename in sorted(os.listdir(tile_label_path)):
        if not filename.endswith(".txt"):
            continue

        label_path = os.path.join(tile_label_path, filename)
        image_path = os.path.join(tile_image_path, filename.replace(".txt", ".jpg"))

        try:
            with open(label_path, "r", encoding="utf-8") as label_file:
                is_empty = not label_file.read().strip()
        except FileNotFoundError:
            log(f"Label file not found: {label_path}")
            continue

        if not is_empty:
            continue

        os.remove(label_path)
        if os.path.exists(image_path):
            os.remove(image_path)
        number_removed += 1

    print(f"Empty tile removal complete. Removed: {number_removed}")

    return number_removed
