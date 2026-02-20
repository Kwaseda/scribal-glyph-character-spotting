# build_class_dictionary(), parse_pseudo_yolo_labels()

# IMPORT CONFIG.PY DATA PROPERLY! - DID IT
import scribal_char_spotting.config as cfg
import json

TXTS_PATH = cfg.TXTS_PATH
COCO_PATH = cfg.COCO_PATH
PSEUDO_YOLO_PATH = cfg.PSEUDO_YOLO_PATH


def build_class_dictionary(COCO_PATH):
    """
    1. Load the full JSON file into memory

    2. Navigate to the "categories" key

    3. FOR EACH category entry:
        Extract: id (integer) and name (letter string)

        Note: COCO IDs start at 1, YOLO class IDs should start at 0
        So: yolo_class_id = category["id"] - 1

        Store pair: letter_string → yolo_class_id
        Example: "a" → 5, "m" → 0, "q" → 1

    4. Also save the reverse: yolo_class_id → letter_string
       (this will be needed for your YOLO .yaml file later)

    5. Save this dictionary somewhere accessible (as a variable or file)

    RETURN: dictionary of letter → integer class_id
    """

    letter_strings = []
    class_ids = []

    letter_dictionary = {}

    with open(COCO_PATH, "r") as file:
        data = json.load(file)

        for category_key in data["categories"]:
            letter_strings.append(category_key["name"])
            class_ids.append(category_key["id"] - 1)

        # print(len(class_ids), len(letter_strings))

        for var in range(len(class_ids)):
            letter_dictionary[letter_strings[var]] = class_ids[var]

        print(letter_dictionary.items())

    # Now, write letter_dictionary as txt to C:\Users\addod\scribal-glyph-character-spotting\txts

    with open(f"{TXTS_PATH}/letter_dictionary.txt", "w") as f:
        # passing an indent parameter makes the json pretty-printed
        json.dump(letter_dictionary, f, indent=2)

    # This loads your dict
    with open(f"{TXTS_PATH}/letter_dictionary.txt", "r") as f:
        my_loaded_dict = json.load(f)

    print(my_loaded_dict)


def parse_pseudo_yolo_labels(label_path, class_dict):
    """
    Takes an image's pseudo YOLO data and letter dictionary

    Parses and stores the pseudo YOLO Label data as an array
    """

    character_array = []
    all_character_array = []

    # Load the letter dictionary for assigning each letter as its officially assigned class_id
    with open(class_dict, "r") as file:
        class_dict = json.load(file)

    # TODO: REMEMBER TO ALTER THIS FILE DIRECTORY SO THAT IT TAKES ALL THE FILES UNDER THE PSEUDO_YOLO_PATH (entire folder)
    # with open(f"{label_path}/WdB_004-0017.txt", "r") as annotation_file:
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


# build_class_dictionary(COCO_PATH)
