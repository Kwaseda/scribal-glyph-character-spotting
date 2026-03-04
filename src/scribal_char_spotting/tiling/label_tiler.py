# filter_labels_for_tile(), normalize_tile_labels()

import numpy as np
import cv2
import scribal_char_spotting.config as cfg

# TODO: Explain filter_labels function to myself and rewrite normalize labels


def filter_labels_for_tile(label_list, tile_coords, tile_size, stride, image):
    """
    Determine which character labels belong to a specific tile based on label center coordinates.
    For tiles at the edges of the image, the tile extends to the full tile_size to cover the
    remaining image space beyond the stride interval.

    Args:
        label_list (list): All character labels for the page as [class_id, x0, y0, w, h]
            where x0, y0 are corner coordinates in pixels and w, h are width and height.
        tile_coords (tuple): (x, y) coordinates for the top-left corner of the tile in pixels.
        tile_size (int): Size of the tile in pixels (square tiles assumed), e.g. 512.
        stride (int): Step size in pixels between consecutive tile positions.
        image (ndarray): The image array from which tile dimensions are derived.

    Returns:
        valid_labels (list): List of labels whose centers fall within the tile boundaries.

    """
    image_height, image_width = image.shape[:2]
    x_min, y_min = tile_coords

    # If last tile in a row, then x_min + stride >= image_width.
    # No more tiles to the right so this tile must claim the remaining space up to tile_size
    if x_min + stride >= image_width:
        x_max = x_min + tile_size
    else:
        x_max = x_min + stride

    # print(f"x_min={x_min}, x_max={x_max}, image_width={image_width}")

    if y_min + stride >= image_height:
        y_max = y_min + tile_size
    else:
        y_max = y_min + stride

    valid_labels = []
    for label in label_list:
        class_id, x0, y0, w, h = label
        # Convert YOLO corner coordinates to center pixel coordinates
        xc = x0 + (w / 2)  # center_x
        yc = y0 + (h / 2)  # center_y

        # Check if the label's center is within the tile
        if x_min <= xc < x_max and y_min <= yc < y_max:
            valid_labels.append([class_id, xc, yc, w, h])

    print(valid_labels)

    return valid_labels


def normalize_tile_labels(tile_label_list, tile_coords, tile_size):
    """
    Converts page-level pixel coordinates for each tile's glyphs into
    YOLO-normalized coordinates relative to the tile.

    Args:
        tile_label_list: list of lists — one list per tile, each containing
                           glyph entries as [class_id, xc, yc, w, h]
        tile_coords:       list of (x, y) top-left pixel corners, one per tile
        tile_size:         pixel size of the tile (square), eg. 512

    Returns:
        List of strings, one per tile, ready to write as a YOLO .txt label file
    """

    new_tile_labels = []

    for tile_idx in range(len(tile_label_list)):
        tile_label = ""
        tile_coord_x = tile_coords[tile_idx][0]
        tile_coord_y = tile_coords[tile_idx][1]

        for label in tile_label_list[tile_idx]:
            class_id, xc, yc, w, h = label

            norm_xc = (xc - tile_coord_x) / tile_size
            norm_yc = (yc - tile_coord_y) / tile_size
            norm_w = w / tile_size
            norm_h = h / tile_size

            tile_label += f"{class_id} {norm_xc} {norm_yc} {norm_w} {norm_h}\n"

        new_tile_labels.append(tile_label)

    # print(new_tile_labels)

    return new_tile_labels
