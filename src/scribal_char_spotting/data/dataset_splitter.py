import scribal_char_spotting.config as cfg
import os, shutil


def make_splits(tile_label_path, tile_image_path):
    train_label_path = cfg.TRAIN_LABELS_PATH
    test_label_path = cfg.TEST_LABELS_PATH
    eval_label_path = cfg.VAL_LABELS_PATH

    train_image_path = cfg.TRAIN_IMAGES_PATH
    test_image_path = cfg.TEST_IMAGES_PATH
    eval_image_path = cfg.VAL_IMAGES_PATH

    os.makedirs(train_label_path, exist_ok=True)
    os.makedirs(test_label_path, exist_ok=True)
    os.makedirs(eval_label_path, exist_ok=True)

    os.makedirs(train_image_path, exist_ok=True)
    os.makedirs(test_image_path, exist_ok=True)
    os.makedirs(eval_image_path, exist_ok=True)

    all_label_files = []
    unique_ids = []
    tile_count = {}

    for txt_filename in sorted(os.listdir(tile_label_path)):
        if txt_filename.endswith(".txt"):
            all_label_files.append(txt_filename)
            unique_id = txt_filename.split("_")[1]
            if unique_id not in unique_ids:
                unique_ids.append(unique_id)
            tile_count[unique_id] = tile_count.get(unique_id, 0) + 1

    # Calculate target tile counts
    total_num_tiles = sum(tile_count.values())
    train_target = total_num_tiles * 0.6
    test_target = total_num_tiles * 0.2
    # val_target   = total_num_tiles * 0.2

    sorted_ids = sorted(unique_ids)

    # Assigning source files to splits

    train_ids = []
    test_ids = []
    val_ids = []

    train_count = 0
    test_count = 0

    # Copy tiles into split folders: train/eval/test
    copied_tiles = 0

    for source_id in sorted_ids:
        file_tile = tile_count[source_id]

        if train_count < train_target:
            train_ids.append(source_id)
            train_count += file_tile

        elif test_count < test_target:
            test_ids.append(source_id)
            test_count += file_tile
        else:

            val_ids.append(source_id)

    for txt_filename in sorted(os.listdir(tile_label_path)):
        source_id = txt_filename.split("_")[1]

        if source_id in train_ids:
            dest_label_path = train_label_path
            dest_image_path = train_image_path
        elif source_id in test_ids:
            dest_label_path = test_label_path
            dest_image_path = test_image_path
        elif source_id in val_ids:
            dest_label_path = eval_label_path
            dest_image_path = eval_image_path
        else:
            continue

        shutil.copy(
            os.path.join(tile_label_path, txt_filename),
            os.path.join(dest_label_path, txt_filename),
        )

        image_filename = txt_filename.replace(".txt", ".jpg")
        image_path = os.path.join(tile_image_path, image_filename)

        # Find that specific image and copy it to the assigned dest path
        if os.path.exists(image_path):
            shutil.copy(image_path, os.path.join(dest_image_path, image_filename))

        copied_tiles += 1
        print(f"{copied_tiles}/{len(all_label_files)}", end="\r", flush=True)

    print(
        f"succesfully copied {copied_tiles} label and image files across train/test/val."
    )


# make_splits(cfg.TILE_LABEL_PATH, cfg.TILE_STORAGE_PATH)


def remove_empty_tiles(tile_label_path, tile_image_path):
    number_removed = 0
    for txt_filename in os.listdir(tile_label_path):
        if txt_filename.endswith(".txt"):
            label_path = os.path.join(tile_label_path, txt_filename)
            image_path = os.path.join(
                tile_image_path, txt_filename.replace(".txt", ".jpg")
            )
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

    print(f"Empty tile removal complete. Removed tiles: {number_removed}")


# remove_empty_tiles(cfg.TILE_LABEL_PATH, cfg.TILE_STORAGE_PATH)
