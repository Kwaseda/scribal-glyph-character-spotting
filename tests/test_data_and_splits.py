"""Augmentation, split construction, and the manifest files YOLO consumes.

The split is the load-bearing one: tiles overlap, so a split that mixes tiles
from one page across train and test leaks the evaluation set into training and
inflates every metric.
"""

import json
import os

import numpy as np
import pytest

import scribal_char_spotting.config as cfg
from scribal_char_spotting.data import (
    blank_tile_regions,
    build_class_dictionary,
    compute_average_background_color,
    make_splits,
    page_book_map,
)
from scribal_char_spotting.utils import generate_split_txts, remove_empty_tiles


# --------------------------------------------------------------------------
# compute_average_background_color
# --------------------------------------------------------------------------


def test_background_colour_comes_from_the_border_not_the_centre():
    image = np.full((64, 64, 3), 200, dtype=np.uint8)
    image[20:44, 20:44] = 0  # a large dark blob the border must ignore

    assert compute_average_background_color(image) == (200, 200, 200)


def test_background_colour_rejects_a_single_channel_image():
    with pytest.raises(ValueError, match="3-channel"):
        compute_average_background_color(np.zeros((10, 10), dtype=np.uint8))


# --------------------------------------------------------------------------
# blank_tile_regions
# --------------------------------------------------------------------------


@pytest.fixture
def glyph_tile():
    image = np.full((64, 64, 3), 200, dtype=np.uint8)
    image[20:30, 20:30] = 0  # the "glyph"
    return image


GLYPH_LABEL = [[0, 25 / 64, 25 / 64, 10 / 64, 10 / 64]]


def test_blank_labeled_erases_the_glyph(glyph_tile):
    result = blank_tile_regions(glyph_tile, GLYPH_LABEL, 64, "blank_labeled")
    assert result[20:30, 20:30].mean() > 150


def test_blank_unlabeled_keeps_the_glyph_and_clears_the_rest(glyph_tile):
    result = blank_tile_regions(glyph_tile, GLYPH_LABEL, 64, "blank_unlabeled")

    assert result[20:30, 20:30].mean() < 50  # glyph preserved
    assert result[0:10, 0:10].mean() > 150  # surroundings blanked


def test_blanking_does_not_modify_the_input(glyph_tile):
    before = glyph_tile.copy()
    blank_tile_regions(glyph_tile, GLYPH_LABEL, 64, "blank_labeled")
    assert np.array_equal(glyph_tile, before)


def test_unknown_mode_is_rejected(glyph_tile):
    with pytest.raises(ValueError, match="mode must be one of"):
        blank_tile_regions(glyph_tile, GLYPH_LABEL, 64, "blank_everything")


def test_a_tile_with_no_labels_is_fully_blanked(glyph_tile):
    result = blank_tile_regions(glyph_tile, [], 64, "blank_unlabeled")
    assert result[20:30, 20:30].mean() > 150


# --------------------------------------------------------------------------
# build_class_dictionary
# --------------------------------------------------------------------------


def test_class_dictionary_shifts_coco_ids_to_zero_indexed(tmp_path, monkeypatch):
    coco = tmp_path / "coco.json"
    coco.write_text(
        json.dumps(
            {"categories": [{"id": 1, "name": "a"}, {"id": 2, "name": "b"}]}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "scribal_char_spotting.data.label_parser.TXTS_PATH", str(tmp_path)
    )

    returned = build_class_dictionary(str(coco), "classes")

    assert returned == {"a": 0, "b": 1}
    written = json.loads((tmp_path / "classes.txt").read_text(encoding="utf-8"))
    assert written == returned, "the file written must match the value returned"


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def _seed_tiles(root, pages):
    """Create label and image files for {page_id: tile_count}."""
    labels = root / "tiled_labels"
    images = root / "tiled_images"
    labels.mkdir()
    images.mkdir()

    for page_id, count in pages.items():
        for tile in range(1, count + 1):
            name = "image_{}_{}".format(page_id, tile)
            (labels / (name + ".txt")).write_text("0 0.5 0.5 0.1 0.1", "utf-8")
            (images / (name + ".jpg")).write_bytes(b"stub")

    return str(labels), str(images)


@pytest.fixture
def split_dirs(tmp_path, monkeypatch):
    out = tmp_path / "dataset"
    paths = {}
    for attr, sub in [
        ("TRAIN_LABELS_PATH", "labels/train"),
        ("VAL_LABELS_PATH", "labels/val"),
        ("TEST_LABELS_PATH", "labels/test"),
        ("TRAIN_IMAGES_PATH", "images/train"),
        ("VAL_IMAGES_PATH", "images/val"),
        ("TEST_IMAGES_PATH", "images/test"),
    ]:
        path = str(out / sub)
        monkeypatch.setattr(cfg, attr, path)
        paths[attr] = path
    return paths


def test_no_page_appears_in_more_than_one_split(tmp_path, split_dirs):
    """The property that keeps overlapping tiles from leaking across splits."""
    pages = {str(i): 10 for i in range(1, 21)}
    labels, images = _seed_tiles(tmp_path, pages)

    assigned = make_splits(labels, images)

    train, test, val = (set(assigned[k]) for k in ("train", "test", "val"))
    assert not train & test
    assert not train & val
    assert not test & val
    assert train | test | val == set(pages)


def test_every_tile_of_a_page_lands_in_the_same_split(tmp_path, split_dirs):
    pages = {str(i): 7 for i in range(1, 13)}
    labels, images = _seed_tiles(tmp_path, pages)

    make_splits(labels, images)

    seen = {}
    for split, path in [
        ("train", split_dirs["TRAIN_LABELS_PATH"]),
        ("test", split_dirs["TEST_LABELS_PATH"]),
        ("val", split_dirs["VAL_LABELS_PATH"]),
    ]:
        for filename in os.listdir(path):
            page_id = filename.split("_")[1]
            seen.setdefault(page_id, set()).add(split)

    for page_id, splits in seen.items():
        assert len(splits) == 1, "page {} spans {}".format(page_id, splits)


def test_split_copies_images_alongside_labels(tmp_path, split_dirs):
    labels, images = _seed_tiles(tmp_path, {str(i): 5 for i in range(1, 11)})

    make_splits(labels, images)

    for label_key, image_key in [
        ("TRAIN_LABELS_PATH", "TRAIN_IMAGES_PATH"),
        ("TEST_LABELS_PATH", "TEST_IMAGES_PATH"),
        ("VAL_LABELS_PATH", "VAL_IMAGES_PATH"),
    ]:
        label_stems = {
            f[:-4] for f in os.listdir(split_dirs[label_key]) if f.endswith(".txt")
        }
        image_stems = {
            f[:-4] for f in os.listdir(split_dirs[image_key]) if f.endswith(".jpg")
        }
        assert label_stems == image_stems


def test_train_receives_the_largest_share(tmp_path, split_dirs):
    labels, images = _seed_tiles(tmp_path, {str(i): 10 for i in range(1, 21)})

    assigned = make_splits(labels, images)

    assert len(assigned["train"]) > len(assigned["test"])
    assert len(assigned["train"]) > len(assigned["val"])


# --------------------------------------------------------------------------
# Manifests and pruning
# --------------------------------------------------------------------------


def test_manifest_is_sorted_and_newline_terminated(tmp_path, split_dirs):
    for key in ("TRAIN_IMAGES_PATH", "VAL_IMAGES_PATH", "TEST_IMAGES_PATH"):
        os.makedirs(split_dirs[key], exist_ok=True)
    for name in ("image_2_1.jpg", "image_10_1.jpg", "image_1_1.jpg"):
        open(os.path.join(split_dirs["TRAIN_IMAGES_PATH"], name), "wb").close()

    generate_split_txts(str(tmp_path / "manifests"))

    text = (tmp_path / "manifests" / "train.txt").read_text(encoding="utf-8")
    lines = text.splitlines()

    assert lines == sorted(lines), "de-tiling re-sorts this; write it sorted"
    assert text.endswith(chr(10)), "a missing final newline undercounts the split"
    assert len(lines) == 3


def test_remove_empty_tiles_drops_the_pair_and_keeps_annotated_ones(tmp_path):
    labels = tmp_path / "labels"
    images = tmp_path / "images"
    labels.mkdir()
    images.mkdir()

    (labels / "image_1_1.txt").write_text("0 0.5 0.5 0.1 0.1", encoding="utf-8")
    (images / "image_1_1.jpg").write_bytes(b"stub")
    (labels / "image_1_2.txt").write_text("   ", encoding="utf-8")
    (images / "image_1_2.jpg").write_bytes(b"stub")

    removed = remove_empty_tiles(str(labels), str(images))

    assert removed == 1
    assert (labels / "image_1_1.txt").exists()
    assert (images / "image_1_1.jpg").exists()
    assert not (labels / "image_1_2.txt").exists()
    assert not (images / "image_1_2.jpg").exists()


# --------------------------------------------------------------------------
# Book-stratified splitting
# --------------------------------------------------------------------------


FOUR_BOOKS = {
    **{str(i): "004" for i in range(1, 11)},
    **{str(i): "006" for i in range(11, 21)},
    **{str(i): "027" for i in range(21, 31)},
    **{str(i): "029" for i in range(31, 35)},
}


def test_page_book_map_reads_the_book_from_scan_filenames(tmp_path):
    for name in ("WdB_004-0029.jpg", "WdB_004-0017.jpg", "WdB_029-0018.jpg"):
        (tmp_path / name).write_bytes(b"stub")

    mapping = page_book_map(str(tmp_path))

    # Pages are numbered by sorted filename order.
    assert mapping == {"1": "004", "2": "004", "3": "029"}


def test_unknown_strategy_is_rejected(tmp_path, split_dirs):
    labels, images = _seed_tiles(tmp_path, {"1": 2, "2": 2})
    with pytest.raises(ValueError, match="strategy must be one of"):
        make_splits(labels, images, strategy="random")


def test_stratified_puts_every_book_in_every_split(tmp_path, split_dirs):
    """The defect the legacy ordering has: validation drawn from one book, and
    two of four books absent from evaluation entirely."""
    labels, images = _seed_tiles(tmp_path, {p: 5 for p in FOUR_BOOKS})

    assigned = make_splits(labels, images, strategy="stratified", books=FOUR_BOOKS)

    all_books = set(FOUR_BOOKS.values())
    for split in ("train", "val", "test"):
        covered = {FOUR_BOOKS[p] for p in assigned[split]}
        assert covered == all_books, "{} misses {}".format(
            split, all_books - covered
        )


def test_stratified_is_still_page_disjoint(tmp_path, split_dirs):
    labels, images = _seed_tiles(tmp_path, {p: 5 for p in FOUR_BOOKS})

    assigned = make_splits(labels, images, strategy="stratified", books=FOUR_BOOKS)

    train, val, test = (set(assigned[k]) for k in ("train", "val", "test"))
    assert not train & val
    assert not train & test
    assert not val & test
    assert train | val | test == set(FOUR_BOOKS)


def test_stratified_keeps_train_the_largest_split(tmp_path, split_dirs):
    labels, images = _seed_tiles(tmp_path, {p: 5 for p in FOUR_BOOKS})

    assigned = make_splits(labels, images, strategy="stratified", books=FOUR_BOOKS)

    assert len(assigned["train"]) > len(assigned["val"])
    assert len(assigned["train"]) > len(assigned["test"])


def test_stratified_is_deterministic(tmp_path, split_dirs):
    """No random seed involved, so two runs must agree exactly."""
    labels, images = _seed_tiles(tmp_path, {p: 3 for p in FOUR_BOOKS})

    first = make_splits(labels, images, strategy="stratified", books=FOUR_BOOKS)
    second = make_splits(labels, images, strategy="stratified", books=FOUR_BOOKS)

    assert first == second


def test_a_two_page_book_stays_in_training_rather_than_being_split(
    tmp_path, split_dirs
):
    """Too few pages to give one to each split without emptying training."""
    books = {"1": "aaa", "2": "aaa", **{str(i): "bbb" for i in range(3, 13)}}
    labels, images = _seed_tiles(tmp_path, {p: 4 for p in books})

    assigned = make_splits(labels, images, strategy="stratified", books=books)

    assert {"1", "2"} <= set(assigned["train"])


def test_legacy_remains_the_default(tmp_path, split_dirs):
    """The committed dataset and every README metric come from the legacy
    ordering, so it must stay reproducible without passing a flag."""
    pages = {str(i): 5 for i in range(1, 21)}
    labels, images = _seed_tiles(tmp_path, pages)

    default = make_splits(labels, images)
    explicit = make_splits(labels, images, strategy="legacy")

    assert default == explicit
