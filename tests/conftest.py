"""Shared fixtures.

Everything here is synthetic. The tests must run on a fresh clone, and the
training data is not redistributable, so nothing may depend on
`data/training-25plus` being present.
"""

import numpy as np
import pytest

TILE_SIZE = 512
OVERLAP = 128
STRIDE = TILE_SIZE - OVERLAP


@pytest.fixture
def tile_params():
    return TILE_SIZE, OVERLAP, STRIDE


@pytest.fixture
def page_image():
    """A 700x900 (WxH) three-channel page of uniform parchment tone."""
    return np.full((900, 700, 3), 210, dtype=np.uint8)


@pytest.fixture
def page_labels():
    """
    Corner-anchored pixel labels, as the pseudo-YOLO annotations supply them.

    Positions are chosen to land in different tiles, including one inside the
    overlap band, and all sit clear of tile edges so a correct round-trip
    reproduces them exactly.
    """
    return [
        [5, 100.0, 100.0, 40.0, 50.0],
        [12, 420.0, 120.0, 30.0, 44.0],   # in the second column's band
        [3, 150.0, 500.0, 36.0, 48.0],    # second row of tiles
        [27, 500.0, 820.0, 28.0, 40.0],   # last tile row
    ]


def pad_targets(width, height, tile_size=TILE_SIZE, stride=STRIDE):
    """The padded dimensions run_pipeline computes for a page."""
    n_x = int(np.ceil(width / stride))
    n_y = int(np.ceil(height / stride))
    return (n_x - 1) * stride + tile_size, (n_y - 1) * stride + tile_size
