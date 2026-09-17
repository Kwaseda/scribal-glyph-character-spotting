"""Map the stratified runs' tile predictions back onto full-page coordinates.

`run_detiling_for_task` builds its paths from config and expects the legacy
layout: predictions under `YOLO_training/predict_project_task_<n>/labels` and
split appears under `data/dataset`. The stratified run writes neither there,
so this script calls `untile_predictions` directly with explicit paths.

Use:
    python scripts/detile_stratified.py            # de-tile only
    python scripts/detile_stratified.py --visualize # also render page overlays
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import scribal_char_spotting.config as cfg
from scribal_char_spotting.tiling.results_detiler import untile_predictions
from scribal_char_spotting.utils import draw_boxes_on_page

DS2 = os.path.join(ROOT, "data", "dataset_stratified")
DS3 = os.path.join(ROOT, "data", "dataset_stratified_task3")
RUNS = os.path.join(ROOT, "YOLO_training", "scribal_runs_stratified")
OUT_BASE = os.path.join(ROOT, "YOLO_training", "results")

# task number -> (split manifest, prediction labels dir)
# Tasks 2 and 3 de-tile over the held-out test split. Task 4 de-tiles over the
# training split, because its question is whether the detector finds characters
# the sparse ground truth never labelled.
TASKS = {
    2: (
        os.path.join(DS2, "test.txt"),
        os.path.join(RUNS, "exp_task2_stratified", "predict_task2_stratified", "labels"),
    ),
    3: (
        os.path.join(DS3, "test.txt"),
        os.path.join(RUNS, "exp_task3_stratified", "predict_task3_stratified", "labels"),
    ),
    4: (
        os.path.join(DS3, "train.txt"),
        os.path.join(RUNS, "exp_task3_stratified", "predict_task4_stratified", "labels"),
    ),
}


def page_filename(image_number):
    """Page N is the Nth scan in sorted order, as run_pipeline numbered them."""
    scans = sorted(f for f in os.listdir(cfg.IMAGE_PATH) if f.endswith(".jpg"))
    return scans[image_number - 1]


def visualize(task_number, output_dir):
    visual_dir = os.path.join(output_dir, f"visualized_task_{task_number}")
    os.makedirs(visual_dir, exist_ok=True)

    written = 0
    for name in sorted(os.listdir(output_dir)):
        if not name.endswith(".txt"):
            continue
        image_number = int(os.path.splitext(name)[0].split("_")[-1])
        scan = os.path.join(cfg.IMAGE_PATH, page_filename(image_number))
        draw_boxes_on_page(
            scan,
            os.path.join(output_dir, name),
            os.path.join(visual_dir, f"page_{image_number}.png"),
        )
        written += 1
    print(f"  rendered {written} page overlays into {visual_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="also render a page overlay per result (slow: matplotlib at 600 dpi)",
    )
    args = parser.parse_args()

    if not os.path.isdir(cfg.IMAGE_PATH):
        raise SystemExit(
            f"{cfg.IMAGE_PATH} not found. De-tiling needs the original scans to "
            "recover true page dimensions. Extract the training-25plus set there."
        )

    for task_number in sorted(TASKS):
        split_txt, labels_dir = TASKS[task_number]

        if not os.path.isdir(labels_dir):
            print(f"task {task_number}: no predictions at {labels_dir}, skipping")
            continue

        output_dir = os.path.join(
            OUT_BASE, f"detiled_predictions_task_{task_number}_stratified"
        )

        n_tiles = sum(1 for line in open(split_txt) if line.strip())
        n_preds = len([f for f in os.listdir(labels_dir) if f.endswith(".txt")])
        print(f"task {task_number}: {n_tiles} tiles listed, {n_preds} prediction files")

        untile_predictions(
            split_txt,
            labels_dir,
            cfg.IMAGE_PATH,
            output_dir,
            cfg.TILE_SIZE,
            cfg.OVERLAP,
        )

        pages = [f for f in os.listdir(output_dir) if f.endswith(".txt")]
        rows = sum(
            sum(1 for line in open(os.path.join(output_dir, p)) if line.strip())
            for p in pages
        )
        print(f"  wrote {len(pages)} pages, {rows} detections -> {output_dir}")

        if args.visualize:
            visualize(task_number, output_dir)


if __name__ == "__main__":
    main()
