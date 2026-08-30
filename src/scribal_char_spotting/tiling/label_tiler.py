"""Assign page-level character labels to tiles and convert them to YOLO format.

Each tile owns a stride-wide band starting at its own origin, so a character is
claimed by exactly one tile: the one whose band contains the character's centre.
The remaining `tile_size - stride` pixels of each tile are overlap, and the
characters there belong to the next tile along.
"""

from scribal_char_spotting.config import log


def filter_labels_for_tile(label_list, tile_coords, tile_size, stride, image):
    """
    Filter character labels to those whose centres fall inside this tile's band.

    Args:
        label_list: list of [class_id, x0, y0, w, h] for all page characters,
                    in pixels, anchored at the upper-left corner
        tile_coords: (x, y) top-left corner of the tile in pixels
        tile_size: pixel size of the tile (square), eg. 512
        stride: step size between consecutive tiles in pixels
        image: padded page array, used to detect the last tile in a row/column

    Returns:
        List of [class_id, xc, yc, w, h] in page pixels, centres converted from
        the corner anchoring used by the source annotations.
    """
    image_height, image_width = image.shape[:2]
    x_min, y_min = tile_coords

    # A tile is last in its row when no further tile fits to the right, i.e.
    # when the next origin (x_min + stride) leaves less than tile_size of image.
    # The last tile claims its full width so the padded margin is still covered.
    if x_min + stride + tile_size > image_width:
        x_max = x_min + tile_size
    else:
        x_max = x_min + stride

    if y_min + stride + tile_size > image_height:
        y_max = y_min + tile_size
    else:
        y_max = y_min + stride

    valid_labels = []
    for label in label_list:
        class_id, x0, y0, w, h = label
        # Source labels are corner-anchored pixels; YOLO wants centres.
        xc = x0 + (w / 2)
        yc = y0 + (h / 2)

        if x_min <= xc < x_max and y_min <= yc < y_max:
            valid_labels.append([class_id, xc, yc, w, h])

    log(f"tile {tile_coords}: {len(valid_labels)} labels")

    return valid_labels


def normalize_tile_labels(tile_label_list, tile_coords, tile_size):
    """
    Convert page-level pixel coordinates into tile-local YOLO-normalised ones.

    Boxes are clipped to the tile. A character whose centre sits near a tile
    edge can extend past it; Ultralytics clips such boxes at load time anyway,
    so clipping here keeps the written files valid YOLO. Exact duplicate rows
    are dropped, because the source annotations contain repeated entries.

    Args:
        tile_label_list: one list per tile of [class_id, xc, yc, w, h] entries
        tile_coords:     list of (x, y) top-left pixel corners, one per tile
        tile_size:       pixel size of the tile (square), eg. 512

    Returns:
        List of strings, one per tile, ready to write as a YOLO .txt label file
    """

    new_tile_labels = []

    for tile_idx in range(len(tile_label_list)):
        tile_coord_x = tile_coords[tile_idx][0]
        tile_coord_y = tile_coords[tile_idx][1]

        rows = []
        seen = set()

        for label in tile_label_list[tile_idx]:
            class_id, xc, yc, w, h = label

            # Tile-local corners, clipped to the tile, then back to centre/size.
            x1 = max(0.0, (xc - w / 2) - tile_coord_x)
            y1 = max(0.0, (yc - h / 2) - tile_coord_y)
            x2 = min(float(tile_size), (xc + w / 2) - tile_coord_x)
            y2 = min(float(tile_size), (yc + h / 2) - tile_coord_y)

            if x2 <= x1 or y2 <= y1:
                # Box lies entirely outside the tile; nothing to write.
                continue

            norm_xc = ((x1 + x2) / 2) / tile_size
            norm_yc = ((y1 + y2) / 2) / tile_size
            norm_w = (x2 - x1) / tile_size
            norm_h = (y2 - y1) / tile_size

            key = (
                class_id,
                round(norm_xc, 9),
                round(norm_yc, 9),
                round(norm_w, 9),
                round(norm_h, 9),
            )
            if key in seen:
                continue
            seen.add(key)

            rows.append(f"{class_id} {norm_xc} {norm_yc} {norm_w} {norm_h}")

        new_tile_labels.append("".join(f"{r}\n" for r in rows))

    return new_tile_labels
