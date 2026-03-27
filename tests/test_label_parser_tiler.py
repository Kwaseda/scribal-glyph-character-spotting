# test build_class_dictionary, parse_pseudo_yolo_labels
# test filter_labels_for_tile, normalize_tile_labels

import os, cv2, json
import numpy as np

from scribal_char_spotting.data import build_class_dictionary, parse_pseudo_yolo_labels
from scribal_char_spotting.tiling import pad_image, get_tile_coords
from scribal_char_spotting.tiling import filter_labels_for_tile, normalize_tile_labels
from scribal_char_spotting.utils import remove_empty_tiles, draw_boxes_on_tile
import scribal_char_spotting.config as cfg

TEST_IMAGE_PATH = os.path.join(cfg.IMAGE_PATH, "WdB_027-0002.jpg")
TEST_LABEL_PATH = os.path.join(cfg.PSEUDO_YOLO_PATH, "WdB_027-0002.txt")
FINAL_LABEL_DIR = cfg.TILE_LABEL_PATH
TXT_PATH = cfg.TXTS_PATH
TESTS_DIR = "C:/Users/addod/scribal-glyph-character-spotting/tests"
letter_dictionary_file = f"{TXT_PATH}/letter_dictionary.txt"

"""Test for parsing the PSEUDO-YOLO label files"""

# build_class_dictionary(cfg.COCO_PATH, "letter_dictionary")

# Parsing the labels

all_labels = parse_pseudo_yolo_labels(TEST_LABEL_PATH, letter_dictionary_file)

print(f"Number of labels:", len(all_labels))
print(f"First 3 labels:", all_labels[:3])


""" Filter labels for a single file, i.e. for all tiles in the padded image """

# Load and pad image
test_image = cv2.imread(TEST_IMAGE_PATH)

h_test, w_test, c_test = test_image.shape
print(f"\nheight:", h_test, ", width:", w_test, ", color_channels", c_test)

""" Testing if padding works properly """
n_tiles_x = int(np.ceil(w_test / cfg.STRIDE))  # horizontal
n_tiles_y = int(np.ceil(h_test / cfg.STRIDE))  # vertical

target_width = (n_tiles_x - 1) * cfg.STRIDE + cfg.TILE_SIZE
target_height = (n_tiles_y - 1) * cfg.STRIDE + cfg.TILE_SIZE

# unpadded = Image.fromarray(test_image).show()

padded_image = pad_image(test_image, target_width, target_height)
# padded = Image.fromarray(padded_image).show()

print(padded_image.shape)

# Get tile coordinates
all_tile_coordinates = get_tile_coords(padded_image, cfg.TILE_SIZE, cfg.OVERLAP)

print(all_tile_coordinates[0], all_tile_coordinates[1])
print(
    f"Number of tiles in total:",
    len(all_tile_coordinates),
)

# Given a tile, filter its corresponding labels

all_tile_labels = []

for i in range(len(all_tile_coordinates)):
    tile_coord = all_tile_coordinates[i]
    tile_label = filter_labels_for_tile(
        all_labels, tile_coord, cfg.TILE_SIZE, cfg.STRIDE, padded_image
    )
    all_tile_labels.append(tile_label)

"""
print(
    f"First tile label: {all_tile_labels[0]}, Number of labels: {len(all_tile_labels)}"
)

print(
    all_tile_labels[:5],
) """

# Verify that all new labels have centers within tile boundaries
for i, tile_label in enumerate(all_tile_labels):
    if tile_label:
        print(f"Tile {i} at {all_tile_coordinates[i]}: {tile_label}")

for i in range(len(all_tile_labels)):
    tile_x, tile_y = all_tile_coordinates[i]

    for tile_label in all_tile_labels[i]:
        print(tile_x, tile_y, tile_label)
    print("x" * 50)


""" Normalize the tile labels"""

normalized_tile_labels = normalize_tile_labels(
    all_tile_labels, all_tile_coordinates, cfg.TILE_SIZE
)

image_name_list = len(
    sorted([f for f in os.listdir(cfg.IMAGE_PATH) if f.endswith(".jpg")])
)
# Write as txt files
for i, label_string in enumerate(normalized_tile_labels):

    output_path = os.path.join(FINAL_LABEL_DIR, f"tile_{image_name_list}_{i+1}.txt")

    with open(output_path, "w") as f:
        f.write(label_string)

print(f"Wrote {len(normalized_tile_labels)} label files to {FINAL_LABEL_DIR}")

""" Removing empty tiles"""
# remove_empty_tiles(cfg.TILE_LABEL_PATH, cfg.TILE_STORAGE_PATH)


letter_txt = os.path.join(cfg.TXTS_PATH, "letter_dictionary.txt")
letter_dictionary = []

with open(letter_txt, "r") as file:
    letter_dictionary = json.load(file)

    print(letter_dictionary.keys())

tile_img_1 = f"{cfg.TILE_STORAGE_PATH}/tile_34_1.jpg"
tile_label_1 = f"{cfg.TILE_LABEL_PATH}/tile_34_1.txt"

# draw_boxes_on_tile(tile_img_1, tile_label_1, letter_dictionary)
