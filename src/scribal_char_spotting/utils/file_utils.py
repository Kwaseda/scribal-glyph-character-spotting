# remove_empty_tiles(), generate_split_txts()
import os
import scribal_char_spotting.config as cfg


def generate_split_txts(txts_path):
    os.makedirs(txts_path, exist_ok=True)

    splits = {
        "train": cfg.TRAIN_IMAGES_PATH,
        "val": cfg.VAL_IMAGES_PATH,
        "test": cfg.TEST_IMAGES_PATH,
    }

    for split_name, split_path in splits.items():
        print(splits)

        jpg_paths = [
            os.path.join(split_path, f)
            for f in os.listdir(split_path)
            if f.endswith(".jpg")
        ]
        txt_file = os.path.join(txts_path, f"{split_name}.txt")
        with open(txt_file, "w") as f:
            f.write("\n".join(jpg_paths))

        print(f"Wrote {len(jpg_paths)} paths to {txt_file}")


def remove_empty_tiles(tile_label_path, tile_image_path):
    number_removed = 0
    for filename in os.listdir(tile_label_path):
        if filename.endswith(".txt"):
            label_path = os.path.join(tile_label_path, filename)
            image_path = os.path.join(tile_image_path, filename.replace(".txt", ".jpg"))
            try:
                with open(label_path, "r") as label_file:
                    is_empty = len(label_file.readlines()) == 0

                if is_empty:
                    os.remove(label_path)
                    if os.path.exists(image_path):
                        # Ensure the image exists before deletion
                        os.remove(image_path)
                        number_removed += 1
            except FileNotFoundError:
                print(f"Label file not found: {label_path}")
                continue

    print(f"Empty tile removal complete. Removed: {number_removed}")


# remove_empty_tiles(cfg.TILE_LABEL_PATH, cfg.TILE_STORAGE_PATH)
