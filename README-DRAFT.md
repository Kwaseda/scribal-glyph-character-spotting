# Scribal Glyph Character Spotting

Character-level detection in 15th-century printed Latin manuscript pages, built around a YOLOv8m
detector with a tiling and de-tiling pipeline.

Data: [ICDAR 2025 / FAU Competition on Glyph Detection in 15th-Century European Printed Documents](https://lme.tf.fau.de/competitions/icdar-2025-competition-on-glyph-detection-in-15th-century-european-printed-documents/),
the `training-25plus` set, limited to books {4, 6, 27, 29}, pages written in minuscule.
29 character classes: a-z plus characters with no modern Latin-English equivalent.

## The problem

Manuscript scans are far too large to feed to a detector directly, and the targets are individual
characters, so they are tiny relative to the page. The pipeline cuts pages into roughly 512x512
slightly overlapping tiles for training, then maps predictions back to full-page coordinates with
non-maximum suppression so results are addressed in page space rather than tile space.

Ground-truth labels arrive as pseudo-YOLO: `[class, x0, y0, w, h]` in **pixels, not normalised**, with
**upper-left corner** coordinates rather than centre coordinates. Converting these correctly, and
re-computing them per tile after cutting, is the step where a silent error would poison everything
downstream. See Tests.

## Tasks and what each one asks

The project runs four tasks. Only the first two produce numbers; that is by design, not by omission.

| Task | Training images | Question | Output |
|---|---|---|---|
| 2 | unmodified tiles | baseline detection quality | metrics |
| 3 | blank out regions where **no** GT is labelled, filled with mean background | how much does surrounding page context contribute? | metrics |
| 4 | blank out regions where GT **is** labelled, filled with mean background, then re-run **inference** | are there character instances the sparse ground truth missed? | qualitative only |

## Measured results

Tasks 2 and 3 are a controlled comparison: same detector, same pipeline, changing only what the
training tiles contain.

| Run | Training tiles | Precision | Recall | mAP50 | mAP50-95 | Epochs |
|---|---|---|---|---|---|---|
| Task 2 | unmodified | 0.653 | 0.812 | **0.761** | **0.721** | 148 |
| Task 3 | non-labelled regions blanked | 0.383 | 0.863 | **0.481** | **0.374** | 150 |

Final-epoch validation metrics from `YOLO_training/exp_train_project_task_*/results.csv`.

**Blanking everything except the labelled characters costs 28 mAP50 points, 0.761 to 0.481.**
Precision falls hardest, 0.653 to 0.383, while recall actually rises slightly. Stripped of
surrounding page context, the detector still finds character-like objects, and marginally more of
them, but is considerably worse at judging which are genuine and which class they belong to. For this
data, context is doing more work for the classifier than for the localiser.

## Task 4: label completeness

Task 4 inverts the manipulation, blanking the labelled regions and re-running inference on those
modified training images. The question is not about context but about the ground truth itself: the
annotated set is sparse, so does the detector surface additional character instances that were never
labelled?

The brief specifies this stage as **qualitative by construction** because no ground truth exists for
the instances in question, so **no numerical result is reported for task 4 and none should be
quoted.** Page-level predictions are in `YOLO_training/results/detiled_predictions_task_4`.

## Pipeline

1. `scripts/run_pipeline.py` reads COCO-format annotations and page scans, cuts overlapping 512x512
   tiles and writes YOLO-format labels with per-tile coordinate recalculation.
2. Train on Colab with `scribal-glyph-charspotting.yaml`.
3. `scripts/run_task2_detiling.py` maps tile predictions back to page coordinates and applies NMS.
4. `scripts/run_augmentations.py` builds the modified training tiles for tasks 3 and 4.
5. `scripts/run_task[3-4]_detiling.py` produce page-level predictions for those tasks.

## Tests

`pytest tests/` covers the two stages where a silent error corrupts every downstream result:
tile geometry, and pseudo-YOLO label parsing across tile boundaries. Given that the source labels are
un-normalised, corner-anchored pixel coordinates that must be converted and then re-cut per tile, a
mis-mapped label produces a detector that trains happily on wrong targets and reports plausible
metrics. These are the two things worth pinning.

## Not implemented

The brief includes two further stages that are not built here:

- **Correlation analysis.** Average the detected glyphs per class across a page or across pages from
  the same book, then correlate that average template against the page. Peaks indicate matches, which
  gives a classical alternative to YOLO where glyph shape varies little.
- **Cross-book bonus.** Repeat tiling, inference and de-tiling on Historia Scholastica pages, to see
  whether the detector transfers to a book it was not trained on.

## Limitations

- Reported metrics are final-epoch validation numbers from the Ultralytics training logs.
- No comparison against a human annotator and no comparison against any prior method on this data,
  so no claim is made about matching or replacing manual expert review.
- Task 4 is qualitative by design; no metric exists for it.
- Trained weights are not included in this repository; training ran on Colab and
  `YOLO_training/saved_models/` is empty.
- Single architecture, single tile size. Neither was ablated.
- Training data is limited, which the brief flags from the outset.

## Requirements

See `requirements.txt`.
