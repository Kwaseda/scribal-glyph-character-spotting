"""

run_detiling.py
Master script for un-tiling YOLO inference results back to page-level detections.
Mirrors the structure of run_pipeline.py.

"""

import os
import scribal_char_spotting.config as cfg
from scribal_char_spotting.tiling import untile_predictions
from scribal_char_spotting.utils import draw_boxes_on_page

original_images_dir = cfg.IMAGE_PATH
test_txt_path = cfg.DATASET_PATH + "/test.txt"

task3_yolo_labels_dir = os.path.join(cfg.YOLO_PATH, "predict_project_task_3", "labels")
detiled_task3_output_dir = os.path.join(
    cfg.YOLO_PATH, "results", "detiled_predictions_task_3"
)

os.makedirs(task3_yolo_labels_dir, exist_ok=True)
os.makedirs(detiled_task3_output_dir, exist_ok=True)

untile_predictions(
    test_txt_path,
    task3_yolo_labels_dir,
    original_images_dir,
    detiled_task3_output_dir,
    cfg.TILE_SIZE,
    cfg.OVERLAP,
)

# Draw boxes on all detiled pages
os.makedirs(os.path.join(detiled_task3_output_dir, "visualized_task_3"), exist_ok=True)

all_original_images = sorted(
    [f for f in os.listdir(original_images_dir) if f.endswith(".jpg")]
)

for result_txt in sorted(os.listdir(detiled_task3_output_dir)):
    if not result_txt.endswith(".txt"):
        continue

    image_number = int(result_txt.replace("image_", "").replace(".txt", ""))
    image_filename = all_original_images[image_number - 1]

    image_path = os.path.join(original_images_dir, image_filename)
    annotation_path = os.path.join(detiled_task3_output_dir, result_txt)
    output_path = os.path.join(
        detiled_task3_output_dir,
        "visualized_task_3",
        result_txt.replace(".txt", "_task3.jpg"),
    )

    draw_boxes_on_page(image_path, annotation_path, output_path)
