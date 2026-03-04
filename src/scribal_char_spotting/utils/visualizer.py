# visualizer.py       ← draw_boxes_on_tile() for verification only
import cv2, os
import scribal_char_spotting.config as cfg
import matplotlib.pyplot as plt
import matplotlib.patches as patches

"""
Pick 5-10 random tiles, run this, look at the output images
If boxes align with actual characters → pipeline is correct
If boxes are offset or wrong size → there is a coordinate bug to find
"""


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
        w_px = w_norm * cfg.TILE_SIZE
        h_px = h_norm * cfg.TILE_SIZE

        # Convert center format to corner format (cast to int for cv2)
        x0 = int(xc_px - w_px / 2)
        y0 = int(yc_px - h_px / 2)
        x1 = int(xc_px + w_px / 2)
        y1 = int(yc_px + h_px / 2)

        # Draw rectangle on image: cv2.rectangle(image, top-left, bottom-right, color, thickness)
        cv2.rectangle(image, (x0 - 5, y0 - 5), (x1 + 5, y1 + 5), (255, 255, 255), 5)

        # Optionally look up the class letter and put text on the image
        if letter_dictionary is not None:
            reverse_dictionary = []
            for key, value in enumerate(letter_dictionary):
                value = letter_dictionary[key]
                reverse_dictionary.append(value)  # FIX HERE

            print(reverse_dictionary)

            # TODO: FIX GLARING REVERSE_DICTIONARY ISSUE WITH NOT SHOWING CLASS, OR NOT

            letter_dict = reverse_dictionary.get(
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
