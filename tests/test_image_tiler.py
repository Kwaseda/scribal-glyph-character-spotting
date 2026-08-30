"""Tile geometry: padding, coordinates, and the cut itself.

A mismatch between the coordinate list and the tiles produced would offset every
label on the page, so these two must be pinned together rather than separately.
"""

import numpy as np
import pytest

from scribal_char_spotting.tiling import (
    get_tile_coords,
    pad_image,
    save_tiles,
    tile_image,
)

from .conftest import pad_targets


def test_pad_image_reaches_target_in_both_axes(page_image):
    target_w, target_h = pad_targets(700, 900)
    padded = pad_image(page_image, target_w, target_h)
    assert padded.shape[:2] == (target_h, target_w)


def test_pad_image_preserves_original_content_at_origin(page_image):
    page_image[5, 7] = (1, 2, 3)
    target_w, target_h = pad_targets(700, 900)
    padded = pad_image(page_image, target_w, target_h)
    assert tuple(padded[5, 7]) == (1, 2, 3)


def test_pad_image_fills_the_margin_with_white(page_image):
    target_w, target_h = pad_targets(700, 900)
    padded = pad_image(page_image, target_w, target_h)
    # Padding is added to the right and bottom only.
    assert (padded[:, 700:] == 255).all()
    assert (padded[900:, :] == 255).all()


def test_pad_image_returns_input_when_already_sized(page_image):
    h, w = page_image.shape[:2]
    assert pad_image(page_image, w, h) is page_image


def test_tile_count_matches_coordinate_count(page_image, tile_params):
    tile_size, overlap, _ = tile_params
    padded = pad_image(page_image, *pad_targets(700, 900))
    coords = get_tile_coords(padded, tile_size, overlap)
    tiles = tile_image(padded, tile_size, overlap)
    assert len(tiles) == len(coords)


def test_tiles_are_square_and_full_size(page_image, tile_params):
    tile_size, overlap, _ = tile_params
    padded = pad_image(page_image, *pad_targets(700, 900))
    for tile in tile_image(padded, tile_size, overlap):
        assert tile.shape[:2] == (tile_size, tile_size)


def test_coordinates_are_ordered_row_major(page_image, tile_params):
    """tile_image iterates y then x; get_tile_coords must agree or every tile
    is paired with the wrong origin."""
    tile_size, overlap, stride = tile_params
    padded = pad_image(page_image, *pad_targets(700, 900))
    coords = get_tile_coords(padded, tile_size, overlap)

    expected = [
        [x, y]
        for y in range(0, padded.shape[0] - tile_size + 1, stride)
        for x in range(0, padded.shape[1] - tile_size + 1, stride)
    ]
    assert coords == expected


def test_each_tile_matches_the_crop_at_its_own_coordinate(page_image, tile_params):
    """The strongest pairing check: tile i must equal the padded page cropped at
    coordinate i."""
    tile_size, overlap, _ = tile_params
    rng = np.random.default_rng(0)
    textured = rng.integers(0, 255, page_image.shape, dtype=np.uint8)
    padded = pad_image(textured, *pad_targets(700, 900))

    coords = get_tile_coords(padded, tile_size, overlap)
    tiles = tile_image(padded, tile_size, overlap)

    for (x, y), tile in zip(coords, tiles):
        assert np.array_equal(tile, padded[y : y + tile_size, x : x + tile_size])


def test_tiles_cover_every_padded_pixel(page_image, tile_params):
    tile_size, overlap, _ = tile_params
    padded = pad_image(page_image, *pad_targets(700, 900))
    covered = np.zeros(padded.shape[:2], dtype=bool)
    for x, y in get_tile_coords(padded, tile_size, overlap):
        covered[y : y + tile_size, x : x + tile_size] = True
    assert covered.all()


@pytest.mark.parametrize("width,height", [(700, 900), (2100, 2800), (2600, 4000)])
def test_padding_never_shrinks_a_page(width, height, tile_params):
    tile_size, overlap, _ = tile_params
    image = np.full((height, width, 3), 200, dtype=np.uint8)
    padded = pad_image(image, *pad_targets(width, height))
    assert padded.shape[0] >= height and padded.shape[1] >= width


def test_save_tiles_writes_one_indexed_files(tmp_path, page_image, tile_params):
    tile_size, overlap, _ = tile_params
    padded = pad_image(page_image, *pad_targets(700, 900))
    tiles = tile_image(padded, tile_size, overlap)

    save_tiles(tiles, 7, str(tmp_path))

    written = sorted(p.name for p in tmp_path.glob("*.jpg"))
    assert written[0] == "image_7_1.jpg"
    assert len(written) == len(tiles)
