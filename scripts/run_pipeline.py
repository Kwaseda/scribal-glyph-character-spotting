# Master script: calls everything in sequence (run_pipeline.py)

"""This package contains your two data modules. A good use of this init is to surface the two functions you will call most often from outside, so your pipeline script has clean imports.
# data/__init__.py
from .label_parser import build_class_dictionary, parse_pseudo_yolo_labels
from .dataset_splitter import make_splits
Now in your pipeline script you can write:
from scribal_spotting.data import build_class_dictionary, make_splits
instead of specifying which file each lives in.
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

"""
At this point, the rest of the pipeline — `pad_image()`, `get_tile_coords()`, `tile_image()`, `save_tiles()`, `valid_tags_normalization()` — is **usable as-is or with only path changes.**

**The main loop pseudo-code (your adapted version of the main block in SegmentationCleaned.py):**
```
SET: tile_size = 512, overlap = 125
LOAD: class_dictionary from COCO JSON

FOR EACH .txt file in your pseudo-YOLO folder:
    
    1. Find corresponding .jpg image file (same name, different extension)
    
    2. Read image with cv2
    
    3. Compute padded width and pad image
    
    4. Generate tile coordinate list
    
    5. Extract tile images
    
    6. Parse annotation file:
       annotations = format_anno_path_scribal(txt_path, class_dictionary)
    
    7. FOR EACH tile coordinate:
       valid_tags = get_valid_tags_scribal(annotations, tile_coords[i], tile_size)
       Append to valid_tag_list
    
    8. Normalize all tile labels:
       new_tags = valid_tags_normalization(valid_tag_list, tile_coords, tile_size)
       (this function is unchanged)
    
    9. Save tile images to output folder
    
    10. Save tile annotation .txt files to output folder
        (one file per tile, empty or not — you'll clean empty ones later)
"""
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

    # TODO: Fix the problem with the YOLO Labels
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

        # TODO: FIX THIS ISSUE WITH ONLY PRINTING OUT ONE TILE COORD

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
