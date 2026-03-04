# Functions: pad_image(), get_tile_coords(), tile_image(), save_tiles()

import numpy as np
import os, cv2

# For testing image modifications
from PIL import Image

import scribal_char_spotting.config as cfg

img_sample_path = f"{cfg.IMAGE_PATH}/WdB_027-0002.jpg"


def pad_image(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Pads an image to the right to match the target width.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        target_width (int): The desired width after padding.

    Returns:
        np.ndarray: The padded image.
    """
    h, w = image.shape[:2]  # Gets me h, w dimensions only, leaving the rest

    right_pad = target_width - w  # For padding only on the right
    bottom_pad = target_height - h  # Padding for on the bottom

    # Order of evaluation in this if matters: w, h => w or h (w,h condition can be true as well)
    if w < target_width and h < target_height:
        padded_image = np.pad(
            image,
            ((0, bottom_pad), (0, right_pad), (0, 0)),
            mode="constant",
            constant_values=255,
        )
        print("Image padded to the right and bottom")

    elif w < target_width:
        padded_image = np.pad(
            image,
            ((0, 0), (0, right_pad), (0, 0)),
            mode="constant",
            constant_values=255,
        )
        print("Image padded to the right")
        return padded_image

    elif h < target_height:
        padded_image = np.pad(
            image,
            ((0, bottom_pad), (0, 0), (0, 0)),
            mode="constant",
            constant_values=255,
        )
        print("Image padded to the bottom")

    else:
        print("No padding needed if already the right size")
        return image

    return padded_image


img_file = np.asarray(Image.open(img_sample_path))

# unpadded = Image.fromarray(img_file).show()

# padded = pad_image(img_file, 3070, 3000)

# padded_img = Image.fromarray(padded).show()


# TODO: TEST THE PADDING WITH ACTUAL IMAGE IN UNTILED IMAGES FOLDER
# Review the Notion lesson on Image tiling first Option B to decide
# the right value for target_width


def get_tile_coords(img_in, tile_sz, overlap):
    """
    Determine pixel coordinates of image tiles.
    Args:
        img_in (numpy array): Input image.
        tile_sz (int): Size of the desired tiles (square).
        overlap (int): Amount of overlap for the tiles.
    Returns:
        tile_coords (list): List of [x, y] start coordinates for each tile.
    """
    imY, imX = img_in.shape[:2]  # Taken as (height, width) instead of w, h
    stride = tile_sz - overlap

    # Calculate the number of tiles that fit into a given image
    nX = len(range(0, imX - tile_sz + 1, stride))
    nY = len(range(0, imY - tile_sz + 1, stride))

    # Determine coordinates
    x_coords = np.arange(0, nX) * stride
    y_coords = np.arange(0, nY) * stride
    XX, YY = np.meshgrid(x_coords, y_coords)
    tile_coords = np.column_stack((XX.ravel(), YY.ravel()))

    return tile_coords.tolist()


# If possible, never use unpadded tile coordinates in the actual final run
# tile_coord_list = get_tile_coords(img_file, cfg.TILE_SIZE, cfg.OVERLAP)
print(" ")
# padded_tile_coord_list = get_tile_coords(padded, cfg.TILE_SIZE, cfg.OVERLAP)


def tile_image(padded_page_image, tile_size, overlap):
    """
    Tiling an image into smaller parts with overlapping regions for object detection.
    """
    # TODO: Fix the bottom-strip problem present where the last row where
    # one or multiple full 512px tile that would extend past the image edge
    # is never created in y and x loop.
    tile_images = []

    height, width = padded_page_image.shape[:2]
    stride = tile_size - overlap

    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            # Extract tile
            tile = padded_page_image[y : y + tile_size, x : x + tile_size]
            tile_images.append(tile)
    return tile_images


"""tiled_images = tile_image(padded, cfg.TILE_SIZE, cfg.OVERLAP)

print(f"Number of tiled images from sample {img_sample_path}:\n ", len(tiled_images))

for img in tiled_images:
    Image.fromarray(img).show() """


def save_tiles(tiles, image_number, output_dir):
    """
    Saves each tile as an image file for visualization.

    Args:
        tiles (list): List of image tiles.
        output_dir (str): Directory where tiles will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    for i, tile in enumerate(tiles):
        tile_filename = os.path.join(output_dir, f"image_{image_number}_{i+1}.jpg")
        cv2.imwrite(tile_filename, tile)
        print(f"Saved: {tile_filename}")


# save_tiles(tiled_images, cfg.TILE_STORAGE_PATH)
