"""Run de-tiling for one task and render the page-level overlays.

Tasks 2 and 3 de-tile predictions over the held-out test split. Task 4 de-tiles
predictions over the training split, because its question is whether the
detector finds characters the sparse ground truth never labelled.
"""

import os

import scribal_char_spotting.config as cfg
from scribal_char_spotting.tiling.results_detiler import untile_predictions
from scribal_char_spotting.utils import draw_boxes_on_page

# Which split each task's predictions were generated over.
TASK_SPLITS = {2: "test", 3: "test", 4: "train"}


def run_detiling_for_task(task_number, visualize=True):
    """
    De-tile one task's tile predictions onto full pages, then draw overlays.

    Args:
        task_number: 2, 3 or 4
        visualize: also render an annotated page image per result

    Returns:
        str: the directory the page-level predictions were written to.
    """
    if task_number not in TASK_SPLITS:
        raise ValueError(
            f"task_number must be one of {sorted(TASK_SPLITS)}, got {task_number}"
        )

    split = TASK_SPLITS[task_number]
    split_txt_path = os.path.join(cfg.DATASET_PATH, f"{split}.txt")

    yolo_labels_dir = os.path.join(
        cfg.YOLO_PATH, f"predict_project_task_{task_number}", "labels"
    )
    output_dir = os.path.join(
        cfg.YOLO_PATH, "results", f"detiled_predictions_task_{task_number}"
    )

    if not os.path.isdir(yolo_labels_dir):
        raise FileNotFoundError(
            f"No tile predictions at {yolo_labels_dir}. Run inference in the "
            "training notebook with save_txt=True before de-tiling."
        )
    if not os.path.exists(split_txt_path):
        raise FileNotFoundError(
            f"No split manifest at {split_txt_path}. Run scripts/run_pipeline.py "
            "first so the tile ordering matches the prediction files."
        )

    os.makedirs(output_dir, exist_ok=True)

    untile_predictions(
        split_txt_path,
        yolo_labels_dir,
        cfg.IMAGE_PATH,
        output_dir,
        cfg.TILE_SIZE,
        cfg.OVERLAP,
    )

    if visualize:
        _visualize_pages(task_number, output_dir)

    return output_dir


def _visualize_pages(task_number, output_dir):
    """Draw each de-tiled page's boxes over its original scan."""
    visual_dir = os.path.join(output_dir, f"visualized_task_{task_number}")
    os.makedirs(visual_dir, exist_ok=True)

    # Page N is the Nth scan in sorted order, the same mapping run_pipeline used.
    original_images = sorted(
        f for f in os.listdir(cfg.IMAGE_PATH) if f.endswith(".jpg")
    )

    for result_txt in sorted(os.listdir(output_dir)):
        if not result_txt.endswith(".txt"):
            continue

        image_number = int(result_txt.replace("image_", "").replace(".txt", ""))
        if not 1 <= image_number <= len(original_images):
            print(f"No original scan for page {image_number}, skipping overlay")
            continue

        draw_boxes_on_page(
            os.path.join(cfg.IMAGE_PATH, original_images[image_number - 1]),
            os.path.join(output_dir, result_txt),
            os.path.join(
                visual_dir, result_txt.replace(".txt", f"_task{task_number}.jpg")
            ),
        )
