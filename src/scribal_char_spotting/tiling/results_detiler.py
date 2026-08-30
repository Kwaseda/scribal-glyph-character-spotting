"""Map tile-level YOLO predictions back onto full-page coordinates.

Tile predictions are de-normalised, offset by their tile's origin on the padded
page, filtered out of the padded margin, and de-duplicated across tile seams
with non-maximum suppression.
"""

import os, cv2
import numpy as np
import scribal_char_spotting.config as cfg
from scribal_char_spotting.tiling import pad_image, get_tile_coords
from scribal_char_spotting.config import log


def parse_tile_prediction_labels(results_label_path):
    """
    Parse YOLO prediction labels from a tile results file.
    Reads bounding box predictions with class IDs, normalized
    coordinates, and confidence scores.
    """
    all_tile_detections = []

    if not os.path.exists(results_label_path):
        return all_tile_detections

    with open(results_label_path, "r") as file:
        label_data = file.readlines()

        for line in label_data:
            parts = line.strip().split()

            if len(parts) < 5:
                continue

            # the tile prediction labels txt has 6 values (original 5 + confidence)
            if len(parts) == 6:
                conf = float(parts[5])
            else:
                conf = 1.0

            # add the detected labels in format: class_id, xc_norm, yc_norm, w_norm, h_norm
            detections = [
                int(parts[0]),
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                conf,
            ]
            all_tile_detections.append(detections)

    return all_tile_detections


def apply_nms_to_page_detections(page_detections, iou_threshold):
    """
    Apply Non-Maximum Suppression to remove duplicate detections across tile boundaries.
    Uses OpenCV's NMS algorithm to filter overlapping boxes based on IoU threshold.
    Returns list of detections that survive the NMS filtering.
    """
    boxes = []
    confidences = []
    class_ids = []

    kept_results = []

    if len(page_detections) == 0:
        return []

    # Convert from center coordinates to top-left corner format for NMS
    for detection in page_detections:

        class_id, xc, yc, w, h, conf = (
            detection[0],
            detection[1],
            detection[2],
            detection[3],
            detection[4],
            detection[5],
        )

        x1 = xc - w / 2
        y1 = yc - h / 2
        boxes.append([x1, y1, w, h])
        confidences.append(conf)
        class_ids.append(class_id)

    # Run OpenCV NMS and collect surviving detections
    kept_indices = cv2.dnn.NMSBoxes(
        boxes, confidences, score_threshold=0.0, nms_threshold=iou_threshold
    )

    for i in np.array(kept_indices).flatten():
        kept_results.append(page_detections[i])

    return kept_results


def denormalize_and_offset_predictions(tile_predictions, tile_origin, tile_size):
    """
    Convert normalized tile-level predictions to page-level pixel coordinates.
    Denormalize coordinates from [0,1] range and offset by tile position on the page.

    Returns detections in absolute page coordinate space.
    """
    x_start, y_start = tile_origin
    page_level_detections = []

    for detection in tile_predictions:

        class_id, xc_norm, yc_norm, w_norm, h_norm, conf = (
            detection[0],
            detection[1],
            detection[2],
            detection[3],
            detection[4],
            detection[5],
        )

        # Convert normalized coordinates to pixels and add tile offset
        xc_px = xc_norm * tile_size + x_start
        yc_px = yc_norm * tile_size + y_start
        w_px = w_norm * tile_size
        h_px = h_norm * tile_size

        page_level_detections.append([class_id, xc_px, yc_px, w_px, h_px, conf])

    return page_level_detections


def untile_predictions(
    test_txt_path,
    test_yolo_labels_dir,
    original_images_dir,
    output_dir,
    tile_size,
    overlap,
    iou_threshold=cfg.IOU_THRESHOLD,
):

    os.makedirs(output_dir, exist_ok=True)

    sorted_tile_paths = []

    with open(test_txt_path, "r") as file:
        data = file.readlines()
        sorted_tile_paths = sorted([line.strip() for line in data])
    #  Build mapping from image_number

    page_groups = {}

    # Ultralytics names prediction files after the position of each image in
    # the list passed to predict(), which the notebook builds with sorted(glob).
    # sorted_tile_paths reproduces that ordering, so index i identifies tile i.
    # Confirm the count matches before relying on it: a silent mismatch would
    # attach every detection to the wrong tile origin without raising.
    n_predictions = len(
        [f for f in os.listdir(test_yolo_labels_dir) if f.endswith(".txt")]
    )
    if n_predictions and n_predictions != len(sorted_tile_paths):
        raise ValueError(
            f"{len(sorted_tile_paths)} tiles listed in {test_txt_path} but "
            f"{n_predictions} prediction files in {test_yolo_labels_dir}. "
            "Predictions are matched to tiles by position, so the counts must "
            "agree or every detection lands on the wrong tile."
        )

    for i, tile_path in enumerate(sorted_tile_paths):

        tile_filename = os.path.basename(tile_path)
        parts = tile_filename.replace(".jpg", "").split("_")
        image_number = int(parts[1])
        tile_index = int(parts[2])

        prediction_path = os.path.join(test_yolo_labels_dir, f"image{i}.txt")

        if image_number not in page_groups:
            page_groups[image_number] = []

        page_groups[image_number].append([tile_index, prediction_path])

    log(f"Found {len(page_groups)} unique pages for detiling")

    # For each page, reconstruct coords, collect detections, NMS, save

    pages_processed = 0
    pages_skipped = 0
    total_detections = 0

    original_images = sorted(
        [f for f in os.listdir(original_images_dir) if f.endswith(".jpg")]
    )

    for image_number, tile_list in page_groups.items():

        # Load original image to get true page dimensions

        image_filename = original_images[image_number - 1]
        orig_image_path = os.path.join(original_images_dir, image_filename)

        if not os.path.exists(orig_image_path):
            print(f"original image not found for image_{image_number}")
            pages_skipped += 1
            continue

        orig_image = cv2.imread(orig_image_path)
        page_height, page_width = orig_image.shape[0], orig_image.shape[1]

        # Reconstruct padded dimensions using same formula as in run_pipeline.py

        n_tiles_x = int(np.ceil(page_width / cfg.STRIDE))
        n_tiles_y = int(np.ceil(page_height / cfg.STRIDE))

        target_width = (n_tiles_x - 1) * cfg.STRIDE + tile_size
        target_height = (n_tiles_y - 1) * cfg.STRIDE + tile_size

        padded_image = pad_image(orig_image, target_width, target_height)
        tile_coords = get_tile_coords(padded_image, tile_size, overlap)

        # Collect detections from all tiles of this page

        all_page_detections = []

        for tile_index, prediction_path in tile_list:

            coord_index = (
                tile_index - 1
            )  # change count base from 1 to 0 (start at index 0)

            if coord_index >= len(tile_coords) or coord_index < 0:
                log(f"coord_index {coord_index} out of range")
                continue

            tile_origin = tile_coords[coord_index]

            tile_preds = parse_tile_prediction_labels(prediction_path)

            if len(tile_preds) == 0:
                continue

            page_preds = denormalize_and_offset_predictions(
                tile_preds, tile_origin, tile_size
            )

            # Drop detections that landed in the padded margin. Padding is
            # added to the right and bottom, so both axes must be checked;
            # testing x alone let bottom-margin detections through.
            page_preds = [
                det
                for det in page_preds
                if det[1] < page_width and det[2] < page_height
            ]

            all_page_detections.extend(page_preds)

        log(f"Detections before NMS: {len(all_page_detections)}")

        # Apply NMS to remove cross-tile duplicates

        final_detections = apply_nms_to_page_detections(
            all_page_detections, iou_threshold=iou_threshold
        )

        log(f"Detections after NMS: {len(final_detections)}")

        # Normalize back to page-level YOLO format and save

        output_path = os.path.join(output_dir, f"image_{image_number}.txt")

        with open(output_path, "w") as f:
            for detection in final_detections:
                class_id, xc_px, yc_px, w_px, h_px, conf = (
                    detection[0],
                    detection[1],
                    detection[2],
                    detection[3],
                    detection[4],
                    detection[5],
                )

                xc_norm = xc_px / page_width
                yc_norm = yc_px / page_height
                w_norm = w_px / page_width
                h_norm = h_px / page_height

                # Keep confidence: it survives NMS and lets downstream
                # consumers threshold without re-running inference.
                f.write(
                    f"{class_id} {xc_norm} {yc_norm} {w_norm} {h_norm} {conf}\n"
                )

        pages_processed += 1
        total_detections += len(final_detections)

    print("=" * 50)
    print("Detiling complete")
    print(f"Pages processed : {pages_processed}")
    print(f"Pages skipped   : {pages_skipped}")
    print(f"Total detections: {total_detections}")
    print(f"Output directory: {output_dir}")
