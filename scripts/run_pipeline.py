# Master script: calls everything in the page tiling sequence (run_pipeline.py)
"""
Script to convert page-level images and pseudo-YOLO labels into
tiled images and normalized tile-level YOLO annotations.

Process:
    1. Load or build class dictionary from the COCO JSON
    2. For each image-label pair:
       - Pad image to tile-compatible dimensions
       - Generate tile coordinates with stride/overlap
       - Extract and save tile images
       - Parse pseudo-YOLO labels
       - Filter labels for each tile
       - Normalize coordinates to tile-local YOLO format
       - Save tile annotations
    3. Remove empty tiles
    4. Split dataset into train/val/test
    5. Generate YOLO format split .txt files
Returns:
    Tiled images and normalized YOLO annotations in output directories
"""

import os, cv2, json
from PIL import Image
import numpy as np


import scribal_char_spotting.config as cfg

from scribal_char_spotting.data import (
    build_class_dictionary,
    parse_pseudo_yolo_labels,
    make_splits,
)
from scribal_char_spotting.tiling import (
    pad_image,
    tile_image,
    get_tile_coords,
    save_tiles,
)
from scribal_char_spotting.tiling import filter_labels_for_tile, normalize_tile_labels
from scribal_char_spotting.utils import remove_empty_tiles, generate_split_txts

TILE_LABEL_DIR = cfg.TILE_LABEL_PATH
TXT_PATH = cfg.TXTS_PATH
TESTS_DIR = "C:/Users/addod/scribal-glyph-character-spotting/tests"
letter_dictionary_file = f"{TXT_PATH}/letter_dictionary.txt"


try:
    with open(letter_dictionary_file, "r") as file:
        letter_dict = json.load(file)
except:
    build_class_dictionary(cfg.COCO_PATH, "letter_dictionary")


all_images = sorted([f for f in os.listdir(cfg.IMAGE_PATH) if f.endswith(".jpg")])
all_labels = sorted([f for f in os.listdir(cfg.PSEUDO_YOLO_PATH) if f.endswith(".txt")])
num_images, num_labels = len(all_images), len(all_labels)

if num_images != num_labels:
    print(f"Number of images: ", num_images, "!= Number of labels: ", num_labels)
else:
    print(f"Equal number of images and labels: {num_images}\n")
    print(f"Now, processing {num_images} images and labels...")

for file_number, filename in enumerate(all_images):
    image_number = file_number + 1

    # IMAGE LOADING AND PADDING

    ORIG_IMAGE_PATH = os.path.join(cfg.IMAGE_PATH, filename)
    print(f"Image path: {ORIG_IMAGE_PATH}")
    print(f"File_number: {file_number, filename}, Image number: {image_number}")

    orig_image = cv2.imread(ORIG_IMAGE_PATH)

    image_height, image_width = orig_image.shape[0], orig_image.shape[1]

    """ Testing if padding works properly """
    n_tiles_x = int(np.ceil(image_width / cfg.STRIDE))  # horizontal
    n_tiles_y = int(np.ceil(image_height / cfg.STRIDE))  # vertical

    target_width = (n_tiles_x - 1) * cfg.STRIDE + cfg.TILE_SIZE
    target_height = (n_tiles_y - 1) * cfg.STRIDE + cfg.TILE_SIZE

    # unpadded = Image.fromarray(test_image).show()

    padded_image = pad_image(orig_image, target_width, target_height)
    # padded = Image.fromarray(padded_image).show()

    print(f"Image padded from {orig_image.shape[:2]} to {padded_image.shape[:2]}")

    # GETTING TILE COORDS AND TILING IMAGE

    tile_coordinates = get_tile_coords(padded_image, cfg.TILE_SIZE, cfg.OVERLAP)

    print("Total number of tiles:", len(tile_coordinates))
    print(f"First 5 tile coords:{tile_coordinates[:5]}")
    print(f"Last 5 tile coords: {tile_coordinates[-5:]}\n")
    print(f"All tile coordinates for image: {tile_coordinates}")

    # Tiling the images

    tot_num_of_tiles = 0

    tiled_images = tile_image(padded_image, cfg.TILE_SIZE, cfg.OVERLAP)

    print(f"Number of tiled images: ", len(tiled_images))

    tot_num_of_tiles += len(tiled_images)

    if len(tiled_images) != len(tile_coordinates):
        print("No, number of tiles don't match the provided no. of coordinates")
    else:
        print("Yes, number of tile match the provided no. of coordinates")

    # Verify the dimensions of the first tile
    first_tile = tiled_images[0].shape
    print(f"Tile dimensions: {first_tile}")

    # SAVE TILED IMAGES TO TILE_STORAGE_PATH

    save_tiles(tiled_images, image_number, cfg.TILE_STORAGE_PATH)

    # Check number of saved files == tiled_images

    num_saved_tiles = len(os.listdir(cfg.TILE_STORAGE_PATH))

    if tot_num_of_tiles != num_saved_tiles:
        print(f"Number of saved tiles != Total num of tiles produced")
    else:
        print("Number of saved tiles and tiles produced are equal")

    ### CONTINUE WITH PARSING THE LABELS FROM NEXT

    label_filename = filename.replace(".jpg", ".txt")
    label_path = os.path.join(cfg.PSEUDO_YOLO_PATH, label_filename)
    print(label_path)

    # Parse and store the YOLO Labels
    all_labels = parse_pseudo_yolo_labels(label_path, letter_dictionary_file)

    print(f"Number of labels:", len(all_labels))
    print(f"First 3 labels:", all_labels[:3])

    # Given an image, filter its corresponding labels for the different tiles

    all_tile_labels = []

    print(f"All tile coords:", tile_coordinates)

    # Iterate over the length of tile_coordinates

    for coord in range(len(tile_coordinates)):
        tile_coord = tile_coordinates[coord]
        print(f"Tile coord:", tile_coord)

        tile_label = filter_labels_for_tile(
            all_labels, tile_coord, cfg.TILE_SIZE, cfg.STRIDE, padded_image
        )
        all_tile_labels.append(tile_label)

        print(f"All tile labels:", all_tile_labels)

        print(f"Tile Coords: {tile_coord}, Tile_label: {tile_label}")
        print("x" * 50)

        # Normalize the tile labels

        normalized_tile_labels = normalize_tile_labels(
            all_tile_labels, tile_coordinates, cfg.TILE_SIZE
        )

        # Write as txt files
        for i, label_string in enumerate(normalized_tile_labels):

            output_path = os.path.join(
                TILE_LABEL_DIR, f"image_{image_number}_{i+1}.txt"
            )

            with open(output_path, "w") as f:
                f.write(label_string)

        print(f"Wrote {len(normalized_tile_labels)} label files to {TILE_LABEL_DIR}")


""" Removing empty tiles"""
remove_empty_tiles(cfg.TILE_LABEL_PATH, cfg.TILE_STORAGE_PATH)


# Split files into datasets
make_splits(cfg.TILE_LABEL_PATH, cfg.TILE_STORAGE_PATH)


# Create YOLO txt files

generate_split_txts(cfg.TILE_LABEL_PATH)
