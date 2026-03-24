import os
import scribal_char_spotting.config as cfg
from scribal_char_spotting.tiling import untile_predictions
from scribal_char_spotting.utils import draw_boxes_on_page

original_images_dir = cfg.IMAGE_PATH
train_txt_path = cfg.DATASET_PATH + "/train.txt"

task4_yolo_labels_dir = os.path.join(cfg.YOLO_PATH, "predict_project_task_4", "labels")
detiled_task4_output_dir = os.path.join(
    cfg.YOLO_PATH, "results", "detiled_predictions_task_4"
)

os.makedirs(task4_yolo_labels_dir, exist_ok=True)
os.makedirs(detiled_task4_output_dir, exist_ok=True)

untile_predictions(
    train_txt_path,
    task4_yolo_labels_dir,
    original_images_dir,
    detiled_task4_output_dir,
    cfg.TILE_SIZE,
    cfg.OVERLAP,
)

# Draw boxes on all detiled pages
os.makedirs(os.path.join(detiled_task4_output_dir, "visualized_task_4"), exist_ok=True)

all_original_images = sorted(
    [f for f in os.listdir(original_images_dir) if f.endswith(".jpg")]
)

for result_txt in sorted(os.listdir(detiled_task4_output_dir)):
    if not result_txt.endswith(".txt"):
        continue

    image_number = int(result_txt.replace("image_", "").replace(".txt", ""))
    image_filename = all_original_images[image_number - 1]

    image_path = os.path.join(original_images_dir, image_filename)
    annotation_path = os.path.join(detiled_task4_output_dir, result_txt)
    output_path = os.path.join(
        detiled_task4_output_dir,
        "visualized_task_4",
        result_txt.replace(".txt", "_task4.jpg"),
    )

    draw_boxes_on_page(image_path, annotation_path, output_path)
