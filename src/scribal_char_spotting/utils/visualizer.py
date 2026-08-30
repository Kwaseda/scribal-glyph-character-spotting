"""Draw ground-truth or predicted boxes over tiles and full pages.

Verification only: if the boxes sit on the characters the coordinate pipeline is
correct, and if they are offset or the wrong size there is a conversion bug.

The class-id -> letter map is loaded lazily, so importing this module (and
therefore the package) does not require the letter dictionary to be present.
"""

import json
import os
import random

import cv2
import matplotlib

matplotlib.use("Agg")  # headless: these functions save files, they do not display
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import scribal_char_spotting.config as cfg

_REVERSE_DICT = None


def _class_names():
    """Load and cache the class-id -> letter mapping, or {} if unavailable."""
    global _REVERSE_DICT
    if _REVERSE_DICT is None:
        try:
            with open(cfg.LETTER_DICTIONARY_PATH, "r", encoding="utf-8") as f:
                letter_dict = json.load(f)
            _REVERSE_DICT = {v: k for k, v in letter_dict.items()}
        except (OSError, json.JSONDecodeError):
            _REVERSE_DICT = {}
    return _REVERSE_DICT


def _label_for(class_id):
    """Human-readable name for a class id, falling back to the id itself."""
    name = _class_names().get(class_id)
    return name if name else str(class_id)


def _read_boxes(annotation_path):
    """Yield (class_id, xc, yc, w, h) from a YOLO label or prediction file."""
    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            yield int(parts[0]), *(float(v) for v in parts[1:5])


def draw_boxes_on_tile(tile_image, tile_annotation_file, output_path=None):
    """
    Draw boxes on a single tile.

    Args:
        tile_image: path to the tile .jpg
        tile_annotation_file: path to its YOLO .txt
        output_path: where to write the annotated tile. If None, the image is
            returned without being written, so this works headless.

    Returns:
        The annotated image array.
    """
    image = cv2.imread(tile_image)
    if image is None:
        raise ValueError(f"Failed to load image: {tile_image}")

    tile_height, tile_width = image.shape[:2]

    for class_id, xc_norm, yc_norm, w_norm, h_norm in _read_boxes(
        tile_annotation_file
    ):
        xc_px, yc_px = xc_norm * tile_width, yc_norm * tile_height
        w_px, h_px = w_norm * tile_width, h_norm * tile_height

        x0, y0 = int(xc_px - w_px / 2), int(yc_px - h_px / 2)
        x1, y1 = int(xc_px + w_px / 2), int(yc_px + h_px / 2)

        cv2.rectangle(image, (x0 - 5, y0 - 5), (x1 + 5, y1 + 5), (0, 255, 0), 2)
        cv2.putText(
            image,
            _label_for(class_id),
            (x0, y0 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, image)

    return image


def draw_boxes_on_page(image_path, annotation_path, output_path, dpi=600):
    """
    Draw boxes on a full page and save the figure.

    Matplotlib rather than cv2 because a page carries thousands of hairline
    boxes and needs vector-quality output to stay legible when zoomed.
    """
    image = np.array(Image.open(image_path))
    image_height, image_width = image.shape[:2]

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.set_aspect("equal")

    for class_id, xc_norm, yc_norm, w_norm, h_norm in _read_boxes(annotation_path):
        box_x = (xc_norm - w_norm / 2) * image_width
        box_y = (yc_norm - h_norm / 2) * image_height

        ax.add_patch(
            patches.Rectangle(
                (box_x, box_y),
                w_norm * image_width,
                h_norm * image_height,
                linewidth=0.2,
                edgecolor=(random.random(), random.random(), random.random()),
                facecolor="none",
            )
        )
        ax.text(
            box_x,
            box_y,
            _label_for(class_id),
            color="black",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.01"),
            fontsize=2,
        )

    ax.axis("off")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)

    print(f"Saved: {output_path}")
