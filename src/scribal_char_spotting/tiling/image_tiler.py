# Functions: pad_image(), get_tile_coords(), tile_image(), save_tiles()

import numpy as np
import os, cv2


def pad_image(image: np.ndarray, target_width: int) -> np.ndarray:
    """Pads an image to the right to match the target width.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        target_width (int): The desired width after padding.

    Returns:
        np.ndarray: The padded image.
    """
    h, w = image.shape[:2]

    if w >= target_width:
        return image  # No padding needed if already the right size

    right_pad = target_width - w  # Padding only on the right
    padded_image = np.pad(
        image, ((0, 0), (0, right_pad), (0, 0)), mode="constant", constant_values=255
    )

    return padded_image


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
    imY, imX = img_in.shape[:2]

    # Calculate the number of tiles that fit into a given image
    nX = int(np.ceil(imX / (tile_sz - overlap)))
    nY = int(np.ceil(imY / (tile_sz - overlap)))

    # Determine coordinates
    x_coords = np.arange(0, nX) * (tile_sz - overlap)
    y_coords = np.arange(0, nY) * (tile_sz - overlap)
    XX, YY = np.meshgrid(x_coords, y_coords)
    tile_coords = np.column_stack((XX.ravel(), YY.ravel()))

    return tile_coords.tolist()


# def tile_image():


def tile_image(padded_page_image, tile_size, overlap):
    """
    Tiling an image into smaller parts with overlapping regions for object detection.
    """
    tile_images = []

    height, width = padded_page_image.shape[:2]
    stride = tile_size - overlap

    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            # Extract tile
            tile = padded_page_image[y : y + tile_size, x : x + tile_size]
            tile_images.append(tile)
    return tile_images


# def save_tiles():


def save_tiles(tiles, form_name, output_dir):
    """
    Saves each tile as an image file for visualization.

    Args:
        tiles (list): List of image tiles.
        output_dir (str): Directory where tiles will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    for i, tile in enumerate(tiles):
        tile_filename = os.path.join(output_dir, f"{form_name}tile_{i}.jpg")
        cv2.imwrite(tile_filename, tile)
        print(f"Saved: {tile_filename}")
