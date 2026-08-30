"""Mapping tile predictions back onto page coordinates.

This is the other place a silent error corrupts everything: a wrong tile origin,
or a prediction bound to the wrong tile, still produces a well-formed page of
boxes that simply sit in the wrong places.
"""

import cv2
import numpy as np
import pytest

from scribal_char_spotting.tiling import (
    apply_nms_to_page_detections,
    denormalize_and_offset_predictions,
    filter_labels_for_tile,
    get_tile_coords,
    normalize_tile_labels,
    pad_image,
    parse_tile_prediction_labels,
    untile_predictions,
)

from .conftest import pad_targets

PAGE_W, PAGE_H = 700, 900


# --------------------------------------------------------------------------
# parse_tile_prediction_labels
# --------------------------------------------------------------------------


def test_missing_prediction_file_reads_as_no_detections(tmp_path):
    assert parse_tile_prediction_labels(str(tmp_path / "absent.txt")) == []


def test_confidence_is_read_when_present(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("3 0.5 0.5 0.1 0.2 0.87", encoding="utf-8")
    assert parse_tile_prediction_labels(str(path)) == [[3, 0.5, 0.5, 0.1, 0.2, 0.87]]


def test_missing_confidence_defaults_to_one(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("3 0.5 0.5 0.1 0.2", encoding="utf-8")
    assert parse_tile_prediction_labels(str(path))[0][5] == 1.0


def test_truncated_rows_are_skipped(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("3 0.5 0.5" + chr(10) + "4 0.1 0.2 0.3 0.4", encoding="utf-8")
    assert len(parse_tile_prediction_labels(str(path))) == 1


# --------------------------------------------------------------------------
# denormalize_and_offset_predictions
# --------------------------------------------------------------------------


def test_denormalise_applies_the_tile_origin():
    (got,) = denormalize_and_offset_predictions(
        [[7, 0.5, 0.25, 0.1, 0.2, 0.9]], (384, 768), 512
    )
    class_id, xc, yc, w, h, conf = got

    assert class_id == 7
    assert xc == pytest.approx(0.5 * 512 + 384)
    assert yc == pytest.approx(0.25 * 512 + 768)
    assert w == pytest.approx(0.1 * 512)
    assert h == pytest.approx(0.2 * 512)
    assert conf == 0.9


def test_denormalise_is_the_inverse_of_normalise():
    tile_origin, tile_size = (384, 768), 512
    original = [[7, 0.5, 0.25, 0.1, 0.2, 1.0]]

    (page,) = denormalize_and_offset_predictions(original, tile_origin, tile_size)
    back = [
        page[0],
        (page[1] - tile_origin[0]) / tile_size,
        (page[2] - tile_origin[1]) / tile_size,
        page[3] / tile_size,
        page[4] / tile_size,
        page[5],
    ]
    assert back == pytest.approx(original[0])


# --------------------------------------------------------------------------
# Non-maximum suppression across tile seams
# --------------------------------------------------------------------------


def test_nms_on_empty_input_returns_empty():
    assert apply_nms_to_page_detections([], 0.45) == []


def test_nms_collapses_a_duplicate_across_a_tile_seam():
    box = [1, 100.0, 100.0, 20.0, 20.0, 0.9]
    near_duplicate = [1, 101.0, 100.0, 20.0, 20.0, 0.7]

    kept = apply_nms_to_page_detections([box, near_duplicate], 0.45)

    assert len(kept) == 1
    assert kept[0][5] == 0.9  # the more confident detection survives


def test_nms_keeps_genuinely_separate_characters():
    a = [1, 100.0, 100.0, 20.0, 20.0, 0.9]
    b = [1, 400.0, 100.0, 20.0, 20.0, 0.8]

    assert len(apply_nms_to_page_detections([a, b], 0.45)) == 2


# --------------------------------------------------------------------------
# Full round trip
# --------------------------------------------------------------------------


def _build_page_fixture(tmp_path, page_labels):
    """Tile a synthetic page, then present its own labels back as predictions."""
    images_dir = tmp_path / "untiled"
    labels_dir = tmp_path / "preds"
    out_dir = tmp_path / "out"
    for directory in (images_dir, labels_dir, out_dir):
        directory.mkdir()

    page = np.full((PAGE_H, PAGE_W, 3), 210, dtype=np.uint8)
    cv2.imwrite(str(images_dir / "page_001.jpg"), page)

    padded = pad_image(page, *pad_targets(PAGE_W, PAGE_H))
    coords = get_tile_coords(padded, 512, 128)

    per_tile = [
        filter_labels_for_tile(page_labels, coord, 512, 384, padded)
        for coord in coords
    ]
    tile_texts = normalize_tile_labels(per_tile, coords, 512)

    # test.txt lists the tiles; prediction files are named by list position.
    manifest = tmp_path / "test.txt"
    manifest.write_text(
        "".join(
            "./images/test/image_1_{}.jpg{}".format(i + 1, chr(10))
            for i in range(len(coords))
        ),
        encoding="utf-8",
    )
    for i, text in enumerate(tile_texts):
        (labels_dir / "image{}.txt".format(i)).write_text(text, encoding="utf-8")

    return manifest, labels_dir, images_dir, out_dir


def test_labels_survive_tiling_and_detiling_back_to_page_coordinates(tmp_path):
    """Feed the tiler output back through de-tiling: the boxes must land where
    they started. This is the end-to-end guard on every coordinate conversion
    in the pipeline."""
    page_labels = [
        [5, 100.0, 100.0, 40.0, 50.0],
        [12, 420.0, 120.0, 30.0, 44.0],
        [3, 150.0, 500.0, 36.0, 48.0],
        [27, 500.0, 820.0, 28.0, 40.0],
    ]
    manifest, labels_dir, images_dir, out_dir = _build_page_fixture(
        tmp_path, page_labels
    )

    untile_predictions(
        str(manifest), str(labels_dir), str(images_dir), str(out_dir), 512, 128
    )

    recovered = []
    for line in (out_dir / "image_1.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        recovered.append(
            (
                int(parts[0]),
                float(parts[1]) * PAGE_W,
                float(parts[2]) * PAGE_H,
                float(parts[3]) * PAGE_W,
                float(parts[4]) * PAGE_H,
            )
        )

    expected = sorted(
        (c, x0 + w / 2, y0 + h / 2, w, h) for c, x0, y0, w, h in page_labels
    )
    assert len(recovered) == len(expected)
    for got, want in zip(sorted(recovered), expected):
        assert got[0] == want[0]
        assert got[1:] == pytest.approx(want[1:], abs=1.0)


def test_detections_in_the_bottom_padding_are_discarded(tmp_path):
    """Padding is added to the right and bottom. A filter that checks only the
    x axis lets bottom-margin detections through into the page results."""
    manifest, labels_dir, images_dir, out_dir = _build_page_fixture(tmp_path, [])

    padded_h = pad_targets(PAGE_W, PAGE_H)[1]
    assert padded_h > PAGE_H, "fixture must actually have bottom padding"

    # The last tile row starts at y=768, so a detection at 0.9 of that tile
    # sits near page-y 1229, far below the real page height of 900.
    for name in ("image4.txt", "image5.txt"):
        (labels_dir / name).write_text("1 0.5 0.9 0.05 0.05 0.99", encoding="utf-8")

    untile_predictions(
        str(manifest), str(labels_dir), str(images_dir), str(out_dir), 512, 128
    )

    result = out_dir / "image_1.txt"
    rows = result.read_text(encoding="utf-8").splitlines() if result.exists() else []
    for line in rows:
        assert float(line.split()[2]) <= 1.0, "detection outside the page: " + line


def test_prediction_count_mismatch_is_rejected(tmp_path):
    """Predictions are bound to tiles by position, so a count mismatch means
    every detection would land on the wrong tile. Fail loudly instead."""
    manifest, labels_dir, images_dir, out_dir = _build_page_fixture(tmp_path, [])
    (labels_dir / "image0.txt").unlink()

    with pytest.raises(ValueError, match="wrong tile"):
        untile_predictions(
            str(manifest), str(labels_dir), str(images_dir), str(out_dir), 512, 128
        )


def test_confidence_is_preserved_in_the_page_output(tmp_path):
    """Confidence survives NMS, so it should reach the file rather than being
    recomputed or dropped."""
    manifest, labels_dir, images_dir, out_dir = _build_page_fixture(tmp_path, [])
    (labels_dir / "image0.txt").write_text(
        "1 0.5 0.5 0.05 0.05 0.63", encoding="utf-8"
    )

    untile_predictions(
        str(manifest), str(labels_dir), str(images_dir), str(out_dir), 512, 128
    )

    (line,) = (out_dir / "image_1.txt").read_text(encoding="utf-8").splitlines()
    assert float(line.split()[5]) == pytest.approx(0.63)
