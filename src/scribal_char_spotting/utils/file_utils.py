# remove_empty_tiles(), generate_split_txts()
import os
import scribal_char_spotting.config as cfg


def generate_split_txts(end_path):
    os.makedirs(end_path, exist_ok=True)

    splits = {
        "train": cfg.TRAIN_IMAGES_PATH,
        "val": cfg.VAL_IMAGES_PATH,
        "test": cfg.TEST_IMAGES_PATH,
    }

    for split_name, split_path in splits.items():

        jpg_paths = [
            f"./images/{split_name}/{f}"
            for f in os.listdir(split_path)
            if f.endswith(".jpg")
        ]
        txt_file = os.path.join(end_path, f"{split_name}.txt")
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


def regenerate_split_txts_for_new_image_dir(
    source_txt_dir, old_image_dir, new_image_dir, output_prefix
):
    """It reads an existing train.txt / val.txt / test.txt, replaces the image directory portion of every
    path with a new one, and writes out new versions like task3_train.txt. Mentioned at the end of Step 3
    in run_augmentation.py as something you could pull out of that script into file_utils.py since it's
    a reusable utility rather than pipeline logic.
    """

    """
    FUNCTION regenerate_split_txts_for_new_image_dir(source_txt_dir, old_image_dir, new_image_dir, output_prefix):

    FOR each split_name in ["train", "val", "test"]:

        source_txt_path = os.path.join(source_txt_dir, f"{split_name}.txt")

        IF source_txt_path does not exist:
            Print "WARNING: {split_name}.txt not found, skipping"
            CONTINUE

        Read all lines from source_txt_path

        new_lines = []
        FOR each line:
            Replace old_image_dir portion of the path with new_image_dir
            Append result to new_lines

        output_filename = f"{output_prefix}_{split_name}.txt"
        Write new_lines to os.path.join(source_txt_dir, output_filename)

        Print "Wrote {len(new_lines)} paths to {output_filename}"
    """

    pass
