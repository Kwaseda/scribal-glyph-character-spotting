"""Label parsing and tile-local conversion.

The source annotations are corner-anchored pixels, not centre-anchored fractions,
and they are re-cut per tile. A mistake in either conversion trains the detector
on wrong targets while still reporting plausible metrics, so these are the two
steps most worth pinning.
"""

import json

import pytest

from scribal_char_spotting.data import parse_pseudo_yolo_labels
from scribal_char_spotting.tiling import (
    filter_labels_for_tile,
    get_tile_coords,
    normalize_tile_labels,
    pad_image,
)

from .conftest import pad_targets

CLASS_DICT = {"a": 5, "m": 0, "zz_ligature with dachförmiges a": 26}


def write_labels(tmp_path, text):
    path = tmp_path / "page.txt"
    path.write_text(text, encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------
# parse_pseudo_yolo_labels
# --------------------------------------------------------------------------


def test_parses_class_name_to_id_and_keeps_pixel_coordinates(tmp_path):
    path = write_labels(tmp_path, "a 10 20 30 40\nm 50 60 12 14\n")
    assert parse_pseudo_yolo_labels(path, CLASS_DICT) == [
        [5, 10.0, 20.0, 30.0, 40.0],
        [0, 50.0, 60.0, 12.0, 14.0],
    ]


def test_parses_class_names_containing_spaces(tmp_path):
    """The ligature class name has spaces; splitting on ' ' would truncate it
    to its first word and fail the dictionary lookup."""
    path = write_labels(tmp_path, "zz_ligature with dachförmiges a 50 60 12 14\n")
    assert parse_pseudo_yolo_labels(path, CLASS_DICT) == [[26, 50.0, 60.0, 12.0, 14.0]]


def test_accepts_a_dictionary_or_a_path_to_one(tmp_path):
    dict_path = tmp_path / "classes.json"
    dict_path.write_text(json.dumps(CLASS_DICT), encoding="utf-8")
    path = write_labels(tmp_path, "a 1 2 3 4\n")
    assert parse_pseudo_yolo_labels(path, str(dict_path)) == [[5, 1.0, 2.0, 3.0, 4.0]]


def test_blank_lines_are_ignored(tmp_path):
    path = write_labels(tmp_path, "a 1 2 3 4\n\n   \nm 5 6 7 8\n")
    assert len(parse_pseudo_yolo_labels(path, CLASS_DICT)) == 2


def test_unknown_class_name_raises_rather_than_guessing(tmp_path):
    path = write_labels(tmp_path, "\u00f8 1 2 3 4\n")
    with pytest.raises(KeyError, match="not in the class dictionary"):
        parse_pseudo_yolo_labels(path, CLASS_DICT)


def test_truncated_line_raises_with_its_line_number(tmp_path):
    path = write_labels(tmp_path, "a 1 2 3 4\nm 5 6\n")
    with pytest.raises(ValueError, match="page.txt:2"):
        parse_pseudo_yolo_labels(path, CLASS_DICT)


# --------------------------------------------------------------------------
# filter_labels_for_tile
# --------------------------------------------------------------------------


def test_filter_converts_corner_anchor_to_centre(page_image, tile_params):
    tile_size, _, stride = tile_params
    padded = pad_image(page_image, *pad_targets(700, 900))
    label = [5, 100.0, 200.0, 40.0, 60.0]

    (got,) = filter_labels_for_tile([label], [0, 0], tile_size, stride, padded)

    assert got == [5, 120.0, 230.0, 40.0, 60.0]  # x0 + w/2, y0 + h/2


def test_every_label_is_claimed_by_exactly_one_tile(
    page_image, page_labels, tile_params
):
    """The property the overlap design depends on. A label claimed twice
    duplicates a character; claimed zero times, it is silently lost."""
    tile_size, overlap, stride = tile_params
    padded = pad_image(page_image, *pad_targets(700, 900))
    coords = get_tile_coords(padded, tile_size, overlap)

    claims = [0] * len(page_labels)
    for tile_coord in coords:
        kept = filter_labels_for_tile(
            page_labels, tile_coord, tile_size, stride, padded
        )
        for label in kept:
            xc, yc = label[1], label[2]
            for i, original in enumerate(page_labels):
                if (
                    xc == original[1] + original[3] / 2
                    and yc == original[2] + original[4] / 2
                ):
                    claims[i] += 1

    assert claims == [1] * len(page_labels)


def test_no_label_position_on_the_page_is_unclaimed(page_image, tile_params):
    """Sweep centres across the whole padded page: each must land in one band."""
    tile_size, overlap, stride = tile_params
    padded = pad_image(page_image, *pad_targets(700, 900))
    coords = get_tile_coords(padded, tile_size, overlap)
    height, width = padded.shape[:2]

    probes = [
        [0, float(x), float(y), 2.0, 2.0]
        for y in range(1, height, 97)
        for x in range(1, width, 89)
    ]

    counts = {i: 0 for i in range(len(probes))}
    for tile_coord in coords:
        for label in filter_labels_for_tile(
            probes, tile_coord, tile_size, stride, padded
        ):
            idx = next(
                i
                for i, p in enumerate(probes)
                if label[1] == p[1] + 1.0 and label[2] == p[2] + 1.0
            )
            counts[idx] += 1

    assert set(counts.values()) == {1}


# --------------------------------------------------------------------------
# normalize_tile_labels
# --------------------------------------------------------------------------


def test_normalises_relative_to_the_tile_origin(tile_params):
    """A tile at x=384 must place a page-x of 640 at 256/512 within the tile."""
    tile_size, _, _ = tile_params
    labels = [[[5, 384.0 + 256.0, 128.0, 64.0, 64.0]]]

    (tile_text,) = normalize_tile_labels(labels, [[384, 0]], tile_size)
    class_id, xc, yc, w, h = tile_text.split()

    assert class_id == "5"
    assert float(xc) == pytest.approx(256 / 512)
    assert float(yc) == pytest.approx(128 / 512)
    assert float(w) == pytest.approx(64 / 512)
    assert float(h) == pytest.approx(64 / 512)


def test_output_is_within_the_unit_square(tile_params):
    """Boxes overhanging a tile edge are clipped, so the written labels are
    valid YOLO rather than relying on the loader to repair them."""
    tile_size, _, _ = tile_params
    labels = [[[5, 10.0, 10.0, 60.0, 60.0]]]  # extends past the tile's origin

    (tile_text,) = normalize_tile_labels(labels, [[0, 0]], tile_size)
    xc, yc, w, h = (float(v) for v in tile_text.split()[1:])

    assert xc - w / 2 >= -1e-9
    assert yc - h / 2 >= -1e-9
    assert xc + w / 2 <= 1 + 1e-9
    assert yc + h / 2 <= 1 + 1e-9


def test_duplicate_rows_are_dropped(tile_params):
    """The source annotations contain repeated entries; Ultralytics silently
    discards them at load time, so remove them where they are written."""
    tile_size, _, _ = tile_params
    duplicated = [[[5, 100.0, 100.0, 20.0, 20.0], [5, 100.0, 100.0, 20.0, 20.0]]]

    (tile_text,) = normalize_tile_labels(duplicated, [[0, 0]], tile_size)

    assert len(tile_text.splitlines()) == 1


def test_distinct_boxes_are_all_kept(tile_params):
    tile_size, _, _ = tile_params
    distinct = [[[5, 100.0, 100.0, 20.0, 20.0], [5, 200.0, 100.0, 20.0, 20.0]]]

    (tile_text,) = normalize_tile_labels(distinct, [[0, 0]], tile_size)

    assert len(tile_text.splitlines()) == 2


def test_box_entirely_outside_the_tile_is_omitted(tile_params):
    tile_size, _, _ = tile_params
    outside = [[[5, -100.0, -100.0, 20.0, 20.0]]]
    assert normalize_tile_labels(outside, [[0, 0]], tile_size) == [""]


def test_one_output_string_per_tile(tile_params):
    """The return value is one label file's contents per tile, in tile order."""
    tile_size, _, _ = tile_params
    labels = [[[5, 100.0, 100.0, 20.0, 20.0]], [], [[0, 500.0, 100.0, 20.0, 20.0]]]

    written = normalize_tile_labels(labels, [[0, 0], [384, 0], [384, 0]], tile_size)

    assert len(written) == 3
    assert written[1] == ""
    assert written[0].startswith("5 ")


def test_every_written_line_ends_with_a_newline(tile_params):
    tile_size, _, _ = tile_params
    labels = [[[5, 100.0, 100.0, 20.0, 20.0], [0, 200.0, 100.0, 20.0, 20.0]]]

    (tile_text,) = normalize_tile_labels(labels, [[0, 0]], tile_size)

    assert tile_text.endswith(chr(10))
    assert len(tile_text.splitlines()) == 2
