"""Split tiles into train/val/test, keeping every tile of a page together.

Tiles overlap by 25%, so two tiles from one page share pixels. Splitting at tile
level would put those shared pixels on both sides of the split and leak the
evaluation set into training. Grouping by source page prevents that.

Two strategies are available.

`legacy` consumes pages in lexicographic id order and fills train, then test,
then validation. It is the strategy that produced the committed dataset and the
metrics in the README, so it is kept byte-for-byte reproducible.

`stratified` allocates each book's pages across the three splits in proportion,
so validation and test both cover every book. Use it for new runs: the legacy
ordering happens to place the whole validation set inside one book and leaves
two of the four books out of evaluation entirely.
"""

import os
import shutil

import scribal_char_spotting.config as cfg

TRAIN_FRACTION = 0.6
TEST_FRACTION = 0.2
VAL_FRACTION = 0.2

STRATEGIES = ("legacy", "stratified")


def page_book_map(image_dir=None):
    """
    Map page number to book id, derived from the original scan filenames.

    Pages are numbered by their position in sorted filename order, the same
    convention `run_pipeline` uses, and the scans are named `WdB_<book>-<folio>`,
    so the book id is recoverable without any extra metadata.

    Returns:
        dict: page number as a string -> book id as a string.
    """
    image_dir = image_dir or cfg.IMAGE_PATH
    scans = sorted(f for f in os.listdir(image_dir) if f.endswith(".jpg"))

    mapping = {}
    for page_number, filename in enumerate(scans, start=1):
        stem = os.path.splitext(filename)[0]
        # WdB_004-0029 -> book "004"
        book = stem.split("_")[-1].split("-")[0] if "_" in stem else stem
        mapping[str(page_number)] = book

    return mapping


def _assign_legacy(tile_count):
    """Fill train to 60% of tiles, then test to 20%, remainder to validation.

    Pages are consumed in lexicographic id order, so page 10 precedes page 2.
    That ordering is load-bearing: it produced the committed dataset and every
    metric in the README, so it must not be changed to a numeric or random sort.
    """
    total = sum(tile_count.values())
    train_target = total * TRAIN_FRACTION
    test_target = total * TEST_FRACTION

    assigned = {"train": [], "test": [], "val": []}
    train_count = test_count = 0

    for page_id in sorted(tile_count):  # lexicographic, deliberately
        page_tiles = tile_count[page_id]
        if train_count < train_target:
            assigned["train"].append(page_id)
            train_count += page_tiles
        elif test_count < test_target:
            assigned["test"].append(page_id)
            test_count += page_tiles
        else:
            assigned["val"].append(page_id)

    return assigned


def _assign_stratified(tile_count, books):
    """Allocate each book's pages across the splits in proportion.

    Every book contributes to validation and test wherever it has enough pages,
    so neither evaluation split is confined to a single book. Within a book,
    pages are taken in numeric order and the first slice goes to train, so the
    result is deterministic and needs no random seed.

    Args:
        tile_count: page id -> number of tiles
        books: page id -> book id

    Returns:
        dict: split name -> list of page ids.
    """
    by_book = {}
    for page_id in tile_count:
        by_book.setdefault(books.get(page_id, "unknown"), []).append(page_id)

    assigned = {"train": [], "test": [], "val": []}

    for book in sorted(by_book):
        pages = sorted(by_book[book], key=lambda p: int(p) if p.isdigit() else p)
        n = len(pages)

        # At least one page each to val and test once a book has three or more,
        # so a small book is not silently dropped from evaluation.
        n_val = max(1, round(n * VAL_FRACTION)) if n >= 3 else 0
        n_test = max(1, round(n * TEST_FRACTION)) if n >= 3 else 0
        if n_val + n_test >= n:
            n_val = n_test = 0 if n < 3 else 1

        n_train = n - n_val - n_test

        assigned["train"].extend(pages[:n_train])
        assigned["val"].extend(pages[n_train : n_train + n_val])
        assigned["test"].extend(pages[n_train + n_val :])

    return assigned


def make_splits(tile_label_path, tile_image_path, strategy="legacy", books=None):
    """
    Copy tiles into train/test/val directories, grouped by source page.

    Args:
        tile_label_path (str): directory of tile .txt labels
        tile_image_path (str): directory of tile .jpg images
        strategy (str): "legacy" reproduces the committed split exactly;
            "stratified" spreads every book across all three splits.
        books (dict): page id -> book id. Required for the stratified strategy;
            read from the original scan filenames when omitted.

    Returns:
        dict: split name -> list of page ids assigned to it.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")

    destinations = {
        "train": (cfg.TRAIN_LABELS_PATH, cfg.TRAIN_IMAGES_PATH),
        "test": (cfg.TEST_LABELS_PATH, cfg.TEST_IMAGES_PATH),
        "val": (cfg.VAL_LABELS_PATH, cfg.VAL_IMAGES_PATH),
    }
    for label_dir, image_dir in destinations.values():
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

    label_files = sorted(f for f in os.listdir(tile_label_path) if f.endswith(".txt"))

    # Tile files are named image_<page>_<tile>.txt, so field 1 is the page id.
    tile_count = {}
    for txt_filename in label_files:
        page_id = txt_filename.split("_")[1]
        tile_count[page_id] = tile_count.get(page_id, 0) + 1

    if strategy == "legacy":
        assigned = _assign_legacy(tile_count)
    else:
        assigned = _assign_stratified(tile_count, books or page_book_map())

    split_of_page = {
        page_id: split for split, pages in assigned.items() for page_id in pages
    }

    copied_tiles = 0
    for txt_filename in label_files:
        split = split_of_page.get(txt_filename.split("_")[1])
        if split is None:
            continue

        label_dir, image_dir = destinations[split]

        shutil.copy(
            os.path.join(tile_label_path, txt_filename),
            os.path.join(label_dir, txt_filename),
        )

        image_filename = txt_filename.replace(".txt", ".jpg")
        image_path = os.path.join(tile_image_path, image_filename)
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(image_dir, image_filename))

        copied_tiles += 1

    total_num_tiles = sum(tile_count.values())
    print(
        f"Copied {copied_tiles} of {len(label_files)} tiles "
        f"across train/test/val using the {strategy} strategy."
    )
    for split in ("train", "val", "test"):
        pages = assigned[split]
        tiles = sum(tile_count[p] for p in pages)
        share = 100 * tiles / total_num_tiles if total_num_tiles else 0
        print(f"  {split:5s}: {len(pages):3d} pages, {tiles:4d} tiles ({share:.1f}%)")

    if books:
        print("  book coverage per split:")
        for split in ("train", "val", "test"):
            counts = {}
            for page_id in assigned[split]:
                book = books.get(page_id, "unknown")
                counts[book] = counts.get(book, 0) + 1
            detail = ", ".join(f"{b}: {c}" for b, c in sorted(counts.items()))
            print(f"    {split:5s}: {detail or 'none'}")

    return assigned
