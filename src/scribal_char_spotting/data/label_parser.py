"""Build the class dictionary from COCO categories and parse pseudo-YOLO labels.

The source annotations are "pseudo-YOLO": whitespace-separated
`<class name> <x0> <y0> <w> <h>` where the coordinates are **pixels, not
normalised**, and anchored at the **upper-left corner**, not the centre.
Both conversions happen downstream in `label_tiler`.
"""

import json
import os

import scribal_char_spotting.config as cfg
from scribal_char_spotting.config import log

TXTS_PATH = cfg.TXTS_PATH
COCO_PATH = cfg.COCO_PATH
PSEUDO_YOLO_PATH = cfg.PSEUDO_YOLO_PATH


def build_class_dictionary(json_path, dict_name):
    """
    Build a name -> class-id mapping from a COCO JSON file and write it to disk.

    COCO category ids are 1-indexed; YOLO class ids are 0-indexed, so each id is
    shifted down by one.

    Args:
        json_path: Path to the COCO format JSON file
        dict_name: Output file stem, written to TXTS_PATH as <dict_name>.txt

    Returns:
        dict: the mapping that was written.
    """
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    letter_dictionary = {}
    for category in data["categories"]:
        # Reversing key and value gives the shape the YOLO yaml expects.
        letter_dictionary[category["name"]] = category["id"] - 1

    os.makedirs(TXTS_PATH, exist_ok=True)
    output_path = os.path.join(TXTS_PATH, f"{dict_name}.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(letter_dictionary, f, indent=2)

    log(f"Wrote {len(letter_dictionary)} classes to {output_path}")

    return letter_dictionary


def parse_pseudo_yolo_labels(label_path, class_dict):
    """
    Read one page's pseudo-YOLO annotations into [class_id, x0, y0, w, h] rows.

    Class names may contain spaces (for example the ligature class), so the five
    fields are taken from the *end* of the line and everything before them is
    rejoined as the name. Splitting on single spaces would truncate such a name
    to its first word and fail the dictionary lookup.

    Args:
        label_path: path to the page's pseudo-YOLO .txt
        class_dict: path to the JSON name -> id mapping, or the mapping itself

    Returns:
        list of [class_id, x0, y0, w, h], coordinates in page pixels.
    """
    if isinstance(class_dict, (str, os.PathLike)):
        with open(class_dict, "r", encoding="utf-8") as file:
            class_dict = json.load(file)

    all_character_array = []

    with open(label_path, "r", encoding="utf-8") as annotation_file:
        characters = annotation_file.readlines()

    for line_number, character in enumerate(characters, start=1):
        parts = character.split()
        if not parts:
            continue
        if len(parts) < 5:
            raise ValueError(
                f"{label_path}:{line_number} has {len(parts)} fields, expected "
                f"at least 5 (<class name> x0 y0 w h): {character.strip()!r}"
            )

        # The last four fields are the box; everything before is the class name.
        letter = " ".join(parts[:-4])
        box = parts[-4:]

        if letter not in class_dict:
            raise KeyError(
                f"{label_path}:{line_number} names class {letter!r}, which is "
                f"not in the class dictionary ({len(class_dict)} entries)."
            )

        all_character_array.append([class_dict[letter]] + [float(v) for v in box])

    log(f"{label_path}: parsed {len(all_character_array)} labels")

    return all_character_array
