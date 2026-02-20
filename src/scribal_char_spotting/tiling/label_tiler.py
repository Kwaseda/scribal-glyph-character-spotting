# filter_labels_for_tile(), normalize_tile_labels()

import numpy as np
import os


def filter_labels_for_tile(tag_list, tile_coords, tile_sz):
    """
    Determine which tags are in a specified tile.
    Args:
        tag_list (list): Ground truth array for all characters [ID, x, y, w, h] (YOLO format).
        tile_coords (tuple): (x, y) coordinates for the top-left corner of the tile.
        tile_sz (int): Size of the tile (square).
    Returns:
        valid_tags (list): List of tags that are within the specified tile.
    """
    x_min, y_min = tile_coords
    x_max, y_max = x_min + tile_sz, y_min + tile_sz

    valid_tags = []
    for tag in tag_list:
        tag_id, tag_x, tag_y, tag_w, tag_h = tag
        # Convert YOLO normalized values back to pixel coordinates
        tag_px = tag_x + tag_w / 2
        tag_py = tag_y + tag_h / 2
        tag_pw, tag_ph = tag_w, tag_h

        # Check if the tag's center is within the tile
        if x_min <= tag_px < x_max and y_min <= tag_py < y_max:
            # valid_tags.append(tag)
            valid_tags.append([tag_id, tag_px, tag_py, tag_pw, tag_ph])

        print(valid_tags)

    return valid_tags


def normalize_tile_labels():
    pass


def valid_tags_normalization(valid_tag_list, tile_coords, tile_size):
    new_tags = []
    for i in range(len(valid_tag_list)):
        new_tags_segment = ""

        for k in range(len(valid_tag_list[i])):
            tag = valid_tag_list[i][k]
            tile_nr = i
            tag_list = tag
            tile_coords_x = tile_coords[i][0]
            tile_coords_y = tile_coords[i][1]

            new_x = (tag_list[1] - tile_coords_x) / tile_size
            new_y = (tag_list[2] - tile_coords_y) / tile_size
            new_width = tag_list[3] / tile_size
            new_height = tag_list[4] / tile_size

            new_tag = f"{tag_list[0]} {new_x} {new_y} {new_width} {new_height}\n"
            new_tags_segment = new_tags_segment + new_tag
        new_tags = new_tags + [new_tags_segment]
    return new_tags
