"""Build the modified training tiles for tasks 3 and 4.

Reads the training tiles and their labels and writes two variants:

    task3/images  everything outside the labelled boxes blanked
    task4/images  the labelled boxes themselves blanked

Only the training split is modified. Validation and test tiles are left as they
are, so a model trained on task 3 tiles is evaluated on unmodified pages.

Run from the repository root:  python scripts/run_augmentations.py
"""

import os

import cv2

import scribal_char_spotting.config as cfg
from scribal_char_spotting.data import blank_tile_regions

SOURCE_IMAGES_DIR = cfg.TRAIN_IMAGES_PATH
SOURCE_LABELS_DIR = cfg.TRAIN_LABELS_PATH
TASK3_IMG_OUTPUT_DIR = os.path.join(cfg.DATASET_PATH, "task3", "images")
TASK4_IMG_OUTPUT_DIR = os.path.join(cfg.DATASET_PATH, "task4", "images")


def parse_tile_labels(label_path):
    """Read a tile's normalised YOLO labels; an absent file means no labels."""
    labels = []
    if not os.path.exists(label_path):
        return labels

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            labels.append([int(parts[0])] + [float(v) for v in parts[1:5]])

    return labels


def main():
    os.makedirs(TASK3_IMG_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TASK4_IMG_OUTPUT_DIR, exist_ok=True)

    all_tile_images = sorted(
        f for f in os.listdir(SOURCE_IMAGES_DIR) if f.endswith(".jpg")
    )
    print(f"Processing {len(all_tile_images)} training tiles...")

    written = skipped = 0

    for image_filename in all_tile_images:
        tile_image = cv2.imread(os.path.join(SOURCE_IMAGES_DIR, image_filename))
        if tile_image is None:
            skipped += 1
            continue

        labels = parse_tile_labels(
            os.path.join(SOURCE_LABELS_DIR, image_filename.replace(".jpg", ".txt"))
        )

        cv2.imwrite(
            os.path.join(TASK3_IMG_OUTPUT_DIR, image_filename),
            blank_tile_regions(tile_image, labels, cfg.TILE_SIZE, "blank_unlabeled"),
        )
        cv2.imwrite(
            os.path.join(TASK4_IMG_OUTPUT_DIR, image_filename),
            blank_tile_regions(tile_image, labels, cfg.TILE_SIZE, "blank_labeled"),
        )
        written += 1

    print(f"\nWrote {written} tiles to each of task3/ and task4/")
    if skipped:
        print(f"Skipped {skipped} unreadable tiles")
    print(f"Task 3 tiles: {TASK3_IMG_OUTPUT_DIR}")
    print(f"Task 4 tiles: {TASK4_IMG_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
