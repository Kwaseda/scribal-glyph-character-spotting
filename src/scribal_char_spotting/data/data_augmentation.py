import numpy as np
import os, cv2, json
from PIL import Image
import scribal_char_spotting.config as cfg

TEST_IMAGE_PATH = os.path.join(cfg.IMAGE_PATH, "WdB_027-0002.jpg")
TEST_LABEL_PATH = os.path.join(cfg.PSEUDO_YOLO_PATH, "WdB_027-0002.txt")

test_image = cv2.imread(TEST_IMAGE_PATH)

test_labels = []


def compute_average_background_color(image):

    border = 10

    # Strip pixels from all four edges:
    top = image[0:border, :]
    bottom = image[-border:, :]
    left = image[:, 0:border]
    right = image[:, -border:]

    top_pixels = top.reshape(-1, 3)  # shape: (10 * image_width, 3)
    bottom_pixels = bottom.reshape(-1, 3)
    left_pixels = left.reshape(-1, 3)
    right_pixels = right.reshape(-1, 3)

    # Stack into one flat array
    # shape: (total_border_pixels, 3)
    all_border_pixels = np.vstack(
        [top_pixels, bottom_pixels, left_pixels, right_pixels]
    )

    # Calculate mean of pixels per channel (B, G, R) and convert to int
    mean_bgr = np.mean(all_border_pixels, axis=0)
    print(mean_bgr)

    tuple_bgr = tuple(mean_bgr.astype(int))

    print(tuple_bgr)

    return tuple_bgr


# avg_bg_color = compute_average_background_color(test_image)


def blank_tile_regions(image, labels, tile_size, mode):

    fill_color = compute_average_background_color(image)
    result = image.copy()

    if mode == "blank_unlabeled":
        result[:] = fill_color  # hide all letters by filling them all with bg color

    for label in labels:

        print(label)
        class_id, xc_norm, yc_norm, w_norm, h_norm = label

        x1 = int((xc_norm - w_norm / 2) * tile_size)
        y1 = int((yc_norm - h_norm / 2) * tile_size)
        x2 = int((xc_norm + w_norm / 2) * tile_size)
        y2 = int((yc_norm + h_norm / 2) * tile_size)

        # Clamp all values to [0, tile_size]
        x1_clamped = np.clip(x1, 0, tile_size)
        y1_clamped = np.clip(y1, 0, tile_size)
        x2_clamped = np.clip(x2, 0, tile_size)
        y2_clamped = np.clip(y2, 0, tile_size)

        if mode == "blank_labeled":
            result[y1_clamped:y2_clamped, x1_clamped:x2_clamped] = (
                fill_color  # hide characters
            )

        elif mode == "blank_unlabeled":
            # TODO: Fix this and understand the code properly
            """
            mask entire image with fill_color first,
            THEN restore result[y1:y2, x1:x2] = image[y1:y2, x1:x2]
            ← keeps only labeled regions, blanks everything else
            """
            result[y1_clamped:y2_clamped, x1_clamped:x2_clamped] = image[
                y1_clamped:y2_clamped, x1_clamped:x2_clamped
            ]

    return result
