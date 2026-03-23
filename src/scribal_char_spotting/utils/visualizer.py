# visualizer.py       ← draw_boxes_on_tile() for verification only
import cv2, os, random, json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import scribal_char_spotting.config as cfg

"""
Pick 5-10 random tiles, run this, look at the output images
If boxes align with actual characters → pipeline is correct
If boxes are offset or wrong size → there is a coordinate bug to find
"""

with open(cfg.TXTS_PATH + "/letter_dictionary.txt", "r") as f:
    letter_dict = json.load(f)

reverse_dict = {}
for key, value in letter_dict.items():
    reverse_dict[value] = key


def draw_boxes_on_tile(tile_image, tile_annotation_file, letter_dictionary=None):

    # Read the image with cv2
    image = cv2.imread(tile_image)
    if image is None:
        raise ValueError(f"Failed to load image: {tile_image}")
    tile_height, tile_width = image.shape[:2]

    # Open and read the annotation file lines
    with open(tile_annotation_file, "r") as f:
        lines = f.readlines()
        print(lines)

    for line in lines:

        # Split the line into parts and unpack: class_id, xc, yc, w, h
        parts = line.strip().split()
        print(parts)
        class_id = int(parts[0])
        xc_norm, yc_norm, w_norm, h_norm = map(float, parts[1:])
        print(f"Parts: {class_id, xc_norm, yc_norm, w_norm, h_norm}\n")

        # Denormalize to pixels
        xc_px = xc_norm * tile_width
        yc_px = yc_norm * tile_height
        w_px = w_norm * tile_width
        h_px = h_norm * tile_height

        # Convert center format to corner format (cast to int for cv2)
        x0 = int(xc_px - w_px / 2)
        y0 = int(yc_px - h_px / 2)
        x1 = int(xc_px + w_px / 2)
        y1 = int(yc_px + h_px / 2)

        # Draw rectangle on image: cv2.rectangle(image, top-left, bottom-right, color, thickness)
        cv2.rectangle(image, (x0 - 5, y0 - 5), (x1 + 5, y1 + 5), (255, 255, 255), 5)

        # Optionally look up the class letter and put text on the image
        if letter_dictionary is not None:

            letter_dict = reverse_dict.get(
                class_id, str(class_id)
            )  # .get(int(class_id), str(class_id))
            print(letter_dict)
            cv2.putText(
                image,
                str(letter_dict),  # FIX THIS PART
                (x0, y0 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

    # Display or save — cv2.imshow or cv2.imwrite
    cv2.imshow("Tile Verification", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


with open(cfg.TXTS_PATH + "/letter_dictionary.txt", "r") as f:
    letter_dict = json.load(f)

all_tile_images = sorted(
    [f for f in os.listdir(cfg.TILE_STORAGE_PATH) if f.endswith(".jpg")]
)
""" sample_tiles = random.sample(all_tile_images, 5)

for tile_filename in sample_tiles:
    tile_image_path = os.path.join(cfg.TILE_STORAGE_PATH, tile_filename)
    tile_label_path = os.path.join(
        cfg.TILE_LABEL_PATH, tile_filename.replace(".jpg", ".txt")
    )

    if not os.path.exists(tile_label_path):
        print(f"No label found for {tile_filename}, skipping")
        continue

    print(f"Showing: {tile_filename}")
    draw_boxes_on_tile(tile_image_path, tile_label_path, letter_dict)

 """


def draw_boxes_on_page(image_path, annotation_path, output_path):

    image = Image.open(image_path)
    image = np.array(image)
    height, width = image.shape[0], image.shape[1]

    with open(annotation_path, "r") as f:
        lines = f.readlines()

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.set_aspect("equal")

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        class_id = int(parts[0])
        xc_norm = float(parts[1])
        yc_norm = float(parts[2])
        w_norm = float(parts[3])
        h_norm = float(parts[4])

        box_x = (xc_norm - w_norm / 2) * width
        box_y = (yc_norm - h_norm / 2) * height
        box_w = w_norm * width
        box_h = h_norm * height

        rect = patches.Rectangle(
            (box_x, box_y),
            box_w,
            box_h,
            linewidth=0.2,
            edgecolor=(random.random(), random.random(), random.random()),
            facecolor="none",
        )
        ax.add_patch(rect)

        letter = reverse_dict.get(class_id, str(class_id))
        ax.text(
            box_x,
            box_y,
            letter,
            color="black",
            ha="center",
            va="center",
            bbox=dict(facecolor="white", edgecolor="none", boxstyle="round,pad=0.01"),
            fontsize=2,
        )

    ax.axis("off")
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0, dpi=1200)
    plt.close(fig)
    print(f"Saved: {output_path}")
