""" 
Re-split the existing tiled dataset without re-tiling from the source scans.
Steps 4 and 5 of the pipeline operate on tiled data; this script reconstructs the page-to-book map
and re-derives the stratified split, verifying against the README.md legacy table.
"""

import collections
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import scribal_char_spotting.config as cfg
from scribal_char_spotting.data.dataset_splitter import make_splits
from scribal_char_spotting.utils import generate_split_txts

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "dataset_stratified")

# Book layout, from README.md: 10 + 10 + 10 + 4 pages in sorted filename order.
BOOK_SIZES = [("004", 10), ("006", 10), ("027", 10), ("029", 4)]


def build_book_map():
    books, page = {}, 1
    for book, count in BOOK_SIZES:
        for _ in range(count):
            books[str(page)] = book
            page += 1
    return books


def verify_against_legacy(books):
    """The reconstructed map must reproduce README.md's legacy book table."""
    expected = {
        "train": {"004": 4, "006": 10, "027": 10, "029": 3},
        "val": {"004": 3},
        "test": {"004": 3, "029": 1},
    }
    for split, want in expected.items():
        pages = {
            os.path.basename(f).split("_")[1]
            for f in glob.glob(os.path.join(cfg.DATASET_LABELS_PATH, split, "*.txt"))
        }
        got = dict(collections.Counter(books[p] for p in pages))
        if got != want:
            raise SystemExit(
                f"Book map does not reproduce the README for {split}: "
                f"expected {want}, got {got}. Refusing to write a dataset."
            )
    print("Book map verified against the README legacy split table.")


def point_config_at(dest):
    cfg.DATASET_PATH = dest
    cfg.DATASET_IMAGES_PATH = os.path.join(dest, "images")
    cfg.DATASET_LABELS_PATH = os.path.join(dest, "labels")
    for split in ("TRAIN", "VAL", "TEST"):
        setattr(cfg, f"{split}_IMAGES_PATH", os.path.join(cfg.DATASET_IMAGES_PATH, split.lower()))
        setattr(cfg, f"{split}_LABELS_PATH", os.path.join(cfg.DATASET_LABELS_PATH, split.lower()))


if __name__ == "__main__":
    books = build_book_map()
    verify_against_legacy(books)

    if os.path.exists(DEST) and any(os.scandir(DEST)):
        raise SystemExit(
            f"{DEST} already exists and is not empty. make_splits copies into its "
            "destination without clearing it, so an existing split would be merged "
            "with the new one and pages would land in two splits at once. Remove the "
            "directory yourself and re-run."
        )

    point_config_at(DEST)
    make_splits(cfg.TILE_LABEL_PATH, cfg.TILE_STORAGE_PATH, strategy="stratified", books=books)
    generate_split_txts(DEST)
    print(f"\nStratified dataset written to {DEST}")
    print("README.md metrics describe the legacy split only. Retrain before quoting anything.")
