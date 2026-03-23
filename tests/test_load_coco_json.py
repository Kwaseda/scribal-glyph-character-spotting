import json

SOURCE_PATH = "C:/Users/addod/scribal-glyph-character-spotting/data"  # Remember: \ won't work, so if you copy directly from the file path, you have to edit it

DATA_PATH = SOURCE_PATH + "/training-25plus"

COCO_PATH = DATA_PATH + "/coco.json"

PSEUDO_YOLO_PATH = DATA_PATH + "/pseudo YOLO"

TILE_STORAGE_PATH = SOURCE_PATH + "/tiled_images"

DATASET_PATH = SOURCE_PATH + "/dataset"


with open(COCO_PATH, "r") as file:
    data = json.load(file)

categories_names = []
image_id_list = []

for image_key in data["images"]:
    image_id_list.append(image_key["id"] - 1)

letter_string_name = []
for category_key in data["categories"]:
    categories_names.append(category_key["name"])
    letter_string_name.append(category_key["id"])

image_info = [
    "image_ids:",
    image_id_list,
    "characters",
    categories_names,
    "letter_string_name",
    letter_string_name,
]


print(f"\n", image_info)
print(json.dumps(data, indent=4))
