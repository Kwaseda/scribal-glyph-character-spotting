# make_splits() logic from newSplitForms.py

# Lines to change: identifier = filename.split("tile")[0]
# png to jpg: shutil.copy(..., filename.replace(".txt", ".jpg"))

import scribal_char_spotting.config as cfg
import os, shutil

tiles_path = cfg.TILE_STORAGE_PATH


def make_splits(forms_compressed_path):
    dataset_path = cfg.DATASET_PATH
    trainset_path = cfg.TRAIN_PATH
    testset_path = cfg.TEST_PATH
    evalset_path = cfg.VAL_PATH

    trainsplit_path = cfg.TRAIN_SPLIT
    testsplit_path = cfg.VAL_SPLIT
    evalsplit_path = cfg.TEST_SPLIT

    os.makedirs(dataset_path, exist_ok=True)
    os.makedirs(trainset_path, exist_ok=True)
    os.makedirs(testset_path, exist_ok=True)
    os.makedirs(evalset_path, exist_ok=True)

    with open(trainsplit_path, "r") as file:
        trainset_identifiers = file.read().splitlines()
    with open(testsplit_path, "r") as file:
        testset_identifiers = file.read().splitlines()
    with open(evalsplit_path, "r") as file:
        evalset_identifiers = file.read().splitlines()

    allforms = os.listdir(forms_compressed_path)
    i = 0

    for filename in allforms:
        if filename.endswith(".txt"):
            i = i + 1
            print(f"{i}/{len(allforms)}", end="\r", flush=True)
            identifier = filename.split("tile")[0]

            if identifier in trainset_identifiers:
                shutil.copy(
                    os.path.join(forms_compressed_path, filename),
                    os.path.join(trainset_path, filename),
                )
                shutil.copy(
                    os.path.join(
                        forms_compressed_path, filename.replace(".txt", ".jpg")
                    ),
                    os.path.join(trainset_path, filename.replace(".txt", ".jpg")),
                )
                continue
            if identifier in testset_identifiers:
                shutil.copy(
                    os.path.join(forms_compressed_path, filename),
                    os.path.join(testset_path, filename),
                )
                shutil.copy(
                    os.path.join(
                        forms_compressed_path, filename.replace(".txt", ".jpg")
                    ),
                    os.path.join(testset_path, filename.replace(".txt", ".jpg")),
                )
                continue
            if identifier in evalset_identifiers:
                shutil.copy(
                    os.path.join(forms_compressed_path, filename),
                    os.path.join(evalset_path, filename),
                )
                shutil.copy(
                    os.path.join(
                        forms_compressed_path, filename.replace(".txt", ".jpg")
                    ),
                    os.path.join(evalset_path, filename.replace(".txt", ".jpg")),
                )
                continue
            i = i - 1
    print(f"succesfully copied {i} files, from current testplit it should be 1199")


make_splits(tiles_path)

# combine_forms(forms_path,forms_compressed_path)
