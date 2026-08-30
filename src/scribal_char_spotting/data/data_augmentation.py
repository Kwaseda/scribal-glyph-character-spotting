"""Build the modified training tiles used by tasks 3 and 4.

Two manipulations, both filling with the tile's own mean border colour so the
replacement blends with the parchment rather than introducing a hard edge:

    blank_unlabeled  keep only the annotated character boxes  (task 3)
    blank_labeled    erase the annotated character boxes      (task 4)
"""

import numpy as np

from scribal_char_spotting.config import log

VALID_MODES = ("blank_labeled", "blank_unlabeled")


def compute_average_background_color(image, border=10):
    """
    Mean BGR colour of the image's border pixels, used as the fill colour.

    Args:
        image: BGR image array
        border: width in pixels of the strip sampled from each edge

    Returns:
        tuple of three ints, the per-channel mean.
    """
    if image.ndim != 3:
        raise ValueError(f"expected a 3-channel image, got shape {image.shape}")

    height, width = image.shape[:2]
    border = max(1, min(border, height, width))

    strips = [
        image[0:border, :],
        image[-border:, :],
        image[:, 0:border],
        image[:, -border:],
    ]
    all_border_pixels = np.vstack([s.reshape(-1, image.shape[2]) for s in strips])

    mean_bgr = np.mean(all_border_pixels, axis=0)
    tuple_bgr = tuple(int(v) for v in mean_bgr)

    log(f"mean border colour: {tuple_bgr}")

    return tuple_bgr


def blank_tile_regions(image, labels, tile_size, mode):
    """
    Blank either the labelled boxes or everything except them.

    Args:
        image: BGR tile array
        labels: [class_id, xc_norm, yc_norm, w_norm, h_norm] rows for this tile
        tile_size: pixel size of the tile (square)
        mode: "blank_labeled" or "blank_unlabeled"

    Returns:
        A new array; the input is not modified.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

    fill_color = compute_average_background_color(image)
    result = image.copy()

    if mode == "blank_unlabeled":
        # Fill everything, then paste the labelled boxes back from the original.
        result[:] = fill_color

    for label in labels:
        _class_id, xc_norm, yc_norm, w_norm, h_norm = label

        x1 = int((xc_norm - w_norm / 2) * tile_size)
        y1 = int((yc_norm - h_norm / 2) * tile_size)
        x2 = int((xc_norm + w_norm / 2) * tile_size)
        y2 = int((yc_norm + h_norm / 2) * tile_size)

        x1 = int(np.clip(x1, 0, tile_size))
        y1 = int(np.clip(y1, 0, tile_size))
        x2 = int(np.clip(x2, 0, tile_size))
        y2 = int(np.clip(y2, 0, tile_size))

        if x2 <= x1 or y2 <= y1:
            # Box clipped away entirely; nothing to blank or restore.
            continue

        if mode == "blank_labeled":
            result[y1:y2, x1:x2] = fill_color
        else:
            result[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    return result
