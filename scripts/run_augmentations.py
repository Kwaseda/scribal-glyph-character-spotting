"""
`run_augmentation.py` needs:
- Train tile images from `cfg.TRAIN_IMAGES_PATH`
- Train tile labels from `cfg.TRAIN_LABELS_PATH`
- Two new output image folders for task3 and task4
"""

import os, cv2
import scribal_char_spotting.config as cfg
from scribal_char_spotting.data import blank_tile_regions

source_images_dir = cfg.TRAIN_IMAGES_PATH
source_labels_dir = cfg.TRAIN_LABELS_PATH
task3_img_output_dir = os.path.join(cfg.DATASET_PATH, "task3/images")
task4_img_output_dir = os.path.join(cfg.DATASET_PATH, "task4/images")


def parse_tile_labels(label_path):
    labels = []
    if not os.path.exists(label_path):
        return labels
    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            labels.append(
                [
                    int(parts[0]),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                ]
            )
    return labels


os.makedirs(task3_img_output_dir, exist_ok=True)
os.makedirs(task4_img_output_dir, exist_ok=True)

all_tile_images = sorted(
    [f for f in os.listdir(source_images_dir) if f.endswith(".jpg")]
)

print(f"Now, processing {len(all_tile_images)} images and labels...")

for image_filename in all_tile_images:

    image_path = os.path.join(source_images_dir, image_filename)
    label_path = os.path.join(source_labels_dir, image_filename.replace(".jpg", ".txt"))

    tile_image = cv2.imread(image_path)

    if tile_image is None:
        continue

    labels = parse_tile_labels(label_path)  # returns [] if label file missing

    print(
        f"Labels:",
        labels,
    )

    task3_image = blank_tile_regions(
        tile_image.copy(), labels, cfg.TILE_SIZE, "blank_unlabeled"
    )
    task4_image = blank_tile_regions(
        tile_image.copy(), labels, cfg.TILE_SIZE, "blank_labeled"
    )

    cv2.imwrite(os.path.join(task3_img_output_dir, image_filename), task3_image)
    cv2.imwrite(os.path.join(task4_img_output_dir, image_filename), task4_image)

num_saved_task3_tiles = len(os.listdir(task3_img_output_dir))
num_saved_task4_tiles = len(os.listdir(task4_img_output_dir))

print(
    f"\nNo. of saved task 3 tiles: {num_saved_task3_tiles,}No. of saved task 4 tiles: {num_saved_task4_tiles}"
)
print(
    f"Task 3 tiles saved to: {task3_img_output_dir} \nTask 4 tiles saved to: {task4_img_output_dir}"
)
