# build_class_dictionary(), parse_pseudo_yolo_labels()

# IMPORT CONFIG.PY DATA PROPERLY! - DID IT
import scribal_char_spotting.config as cfg
import json

TXTS_PATH = cfg.TXTS_PATH
COCO_PATH = cfg.COCO_PATH
PSEUDO_YOLO_PATH = cfg.PSEUDO_YOLO_PATH


def build_class_dictionary(json_path, dict_name):
    """
    Builds a dictionary mapping character names to class IDs from a COCO JSON file.

    Args:
        json_path: Path to the COCO format JSON file
        dict_name: Output dictionary file name

    Returns:
        Dictionary mapping letter strings to YOLO class IDs (0-indexed)
    """

    letter_strings = []
    class_ids = []

    letter_dictionary = {}

    with open(json_path, "r") as file:
        data = json.load(file)

        for category_key in data["categories"]:
            letter_strings.append(category_key["name"])
            class_ids.append(category_key["id"] - 1)

        # print(len(class_ids), len(letter_strings))

        for var in range(len(class_ids)):
            # Reversing placement of key and value creates the type easily associated with YOLO-yaml file
            letter_dictionary[letter_strings[var]] = class_ids[var]

    # Now, write letter_dictionary as txt to C:\Users\addod\scribal-glyph-character-spotting\txts

    with open(f"{TXTS_PATH}/{dict_name}.txt", "w") as f:
        # passing an indent parameter makes the json pretty-printed
        json.dump(letter_dictionary, f, indent=2)

    # This loads the dict
    with open(f"{TXTS_PATH}/letter_dictionary.txt", "r") as f:
        letter_dict = json.load(f)

    print(letter_dict)


def parse_pseudo_yolo_labels(label_path, class_dict):
    """
    Takes an image's pseudo YOLO data and letter dictionary

    Parses and stores the pseudo YOLO Label data as an array
    """

    all_character_array = []

    # Load the letter dictionary for assigning each letter as its officially assigned class_id
    with open(class_dict, "r") as file:
        class_dict = json.load(file)

    with open(label_path, "r") as annotation_file:
        characters = annotation_file.readlines()
    for character in characters:
        character_array = []
        data = character.split(" ")
        for i in range(5):
            if i == 0:
                letter = data[i]
                class_id = class_dict[letter]  # look up the letter in your dictionary
                character_array.append(class_id)

            else:
                character_array.append(float(data[i]))
        all_character_array.append(character_array)
    print(all_character_array)

    return all_character_array


letter_dictionary_file = f"{TXTS_PATH}/letter_dictionary.txt"


# parse_pseudo_yolo_labels(PSEUDO_YOLO_PATH, letter_dictionary_file)


# build_class_dictionary(COCO_PATH, "letter_dict_yaml")
