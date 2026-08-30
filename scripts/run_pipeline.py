"""Turn page scans and pseudo-YOLO annotations into a tiled YOLO dataset.

Steps:
    1. Load, or build from the COCO JSON, the class dictionary
    2. For each page:
       - pad to tile-compatible dimensions
       - generate tile coordinates at the configured stride
       - cut and save the tiles
       - parse the page's pseudo-YOLO labels
       - assign each label to the tile owning its centre
       - convert to tile-local normalised YOLO coordinates and write them
    3. Remove tiles with no annotations
    4. Split into train/val/test, keeping every tile of a page in one split
    5. Write the YOLO manifests

Run from the repository root:

    python scripts/run_pipeline.py                      # legacy split
    python scripts/run_pipeline.py --split stratified   # every book in every split

The legacy split reproduces the committed dataset and the metrics in the README.
The stratified split spreads each book across train, validation and test, which
the legacy ordering does not: it places the whole validation set inside one book
and leaves two of the four books out of evaluation entirely. Re-running with
--split stratified changes the dataset, so the model must be retrained and the
reported metrics re-measured before they mean anything.
"""

import argparse
import json
import os

import cv2
import numpy as np

import scribal_char_spotting.config as cfg
from scribal_char_spotting.data import (
    build_class_dictionary,
    make_splits,
    page_book_map,
    parse_pseudo_yolo_labels,
)
from scribal_char_spotting.tiling import (
    filter_labels_for_tile,
    get_tile_coords,
    normalize_tile_labels,
    pad_image,
    save_tiles,
    tile_image,
)
from scribal_char_spotting.utils import generate_split_txts, remove_empty_tiles

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--split",
    choices=("legacy", "stratified"),
    default="legacy",
    help=(
        "legacy reproduces the committed dataset and the README metrics; "
        "stratified spreads every book across all three splits"
    ),
)
args = parser.parse_args()

TILE_LABEL_DIR = cfg.TILE_LABEL_PATH
LETTER_DICTIONARY_FILE = cfg.LETTER_DICTIONARY_PATH

os.makedirs(TILE_LABEL_DIR, exist_ok=True)
os.makedirs(cfg.TILE_STORAGE_PATH, exist_ok=True)

try:
    with open(LETTER_DICTIONARY_FILE, "r", encoding="utf-8") as file:
        letter_dict = json.load(file)
except (OSError, json.JSONDecodeError):
    letter_dict = build_class_dictionary(cfg.COCO_PATH, "letter_dictionary")

print(f"Class dictionary: {len(letter_dict)} classes")

image_filenames = sorted(f for f in os.listdir(cfg.IMAGE_PATH) if f.endswith(".jpg"))
label_filenames = sorted(
    f for f in os.listdir(cfg.PSEUDO_YOLO_PATH) if f.endswith(".txt")
)

if len(image_filenames) != len(label_filenames):
    print(
        f"Warning: {len(image_filenames)} images but {len(label_filenames)} label "
        "files. Pages without a matching label file will be skipped."
    )
else:
    print(f"Processing {len(image_filenames)} images and labels...")

tiles_written = 0
labels_written = 0

for file_number, filename in enumerate(image_filenames):
    # Page numbers are 1-based and index into sorted(image_filenames); the
    # de-tiling step reconstructs the same mapping, so the sort must not change.
    image_number = file_number + 1

    orig_image_path = os.path.join(cfg.IMAGE_PATH, filename)
    orig_image = cv2.imread(orig_image_path)
    if orig_image is None:
        print(f"Skipping unreadable image: {orig_image_path}")
        continue

    label_path = os.path.join(cfg.PSEUDO_YOLO_PATH, filename.replace(".jpg", ".txt"))
    if not os.path.exists(label_path):
        print(f"Skipping {filename}: no matching label file")
        continue

    image_height, image_width = orig_image.shape[:2]

    # Pad up so a whole number of tiles fits, which also guarantees that every
    # real page pixel falls inside some tile's claimed band.
    n_tiles_x = int(np.ceil(image_width / cfg.STRIDE))
    n_tiles_y = int(np.ceil(image_height / cfg.STRIDE))
    target_width = (n_tiles_x - 1) * cfg.STRIDE + cfg.TILE_SIZE
    target_height = (n_tiles_y - 1) * cfg.STRIDE + cfg.TILE_SIZE

    padded_image = pad_image(orig_image, target_width, target_height)

    tile_coordinates = get_tile_coords(padded_image, cfg.TILE_SIZE, cfg.OVERLAP)
    tiled_images = tile_image(padded_image, cfg.TILE_SIZE, cfg.OVERLAP)

    if len(tiled_images) != len(tile_coordinates):
        raise RuntimeError(
            f"{filename}: {len(tiled_images)} tiles but "
            f"{len(tile_coordinates)} coordinates. Tiling and coordinate "
            "generation have diverged; every label would land on the wrong tile."
        )

    save_tiles(tiled_images, image_number, cfg.TILE_STORAGE_PATH)
    tiles_written += len(tiled_images)

    page_labels = parse_pseudo_yolo_labels(label_path, letter_dict)

    # Assign every label to the tile that owns its centre, then convert the whole
    # page in one pass. Normalising inside this loop would redo the work, and
    # rewrite every file, once per tile.
    all_tile_labels = [
        filter_labels_for_tile(
            page_labels, tile_coord, cfg.TILE_SIZE, cfg.STRIDE, padded_image
        )
        for tile_coord in tile_coordinates
    ]

    normalized_tile_labels = normalize_tile_labels(
        all_tile_labels, tile_coordinates, cfg.TILE_SIZE
    )

    for i, label_string in enumerate(normalized_tile_labels):
        output_path = os.path.join(TILE_LABEL_DIR, f"image_{image_number}_{i+1}.txt")
        with open(output_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(label_string)

    labels_written += len(normalized_tile_labels)

    print(
        f"[{image_number}/{len(image_filenames)}] {filename}: "
        f"{orig_image.shape[:2]} -> {padded_image.shape[:2]}, "
        f"{len(tiled_images)} tiles, {len(page_labels)} labels"
    )

print(f"\nTiling complete: {tiles_written} tiles, {labels_written} label files")

remove_empty_tiles(cfg.TILE_LABEL_PATH, cfg.TILE_STORAGE_PATH)

books = page_book_map(cfg.IMAGE_PATH) if args.split == "stratified" else None
make_splits(
    cfg.TILE_LABEL_PATH, cfg.TILE_STORAGE_PATH, strategy=args.split, books=books
)

generate_split_txts(cfg.DATASET_PATH)

if args.split == "stratified":
    print(
        "",
        "This dataset uses the stratified split, so the metrics in "
        "README.md no longer describe it. Retrain before quoting any "
        "number.",
        sep=chr(10),
    )
