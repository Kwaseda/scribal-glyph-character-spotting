# Scribal glyph character spotting

Character-level object detection on 15th-century printed Latin pages, built around a YOLOv8m
detector with a tiling and de-tiling pipeline.

Data comes from the [ICDAR 2025 / FAU competition on glyph detection in 15th-century European
printed documents](https://lme.tf.fau.de/competitions/icdar-2025-competition-on-glyph-detection-in-15th-century-european-printed-documents/),
the `training-25plus` set, restricted to books {4, 6, 27, 29} and to pages set in minuscule. These
are incunabula, early printed books rather than handwritten manuscripts, so glyph shapes repeat far
more consistently than scribal hands would.

## The problem

Page scans are too large to feed to a detector directly, and the targets are individual characters,
which are tiny relative to the page. The pipeline cuts each page into 512x512 tiles overlapping by
128 pixels, trains on those, then maps predictions back into full-page coordinates with non-maximum
suppression.

Ground-truth labels arrive in a pseudo-YOLO format: `<class name> x0 y0 w h`, where the coordinates
are pixels rather than normalised fractions and are anchored at the upper-left corner rather than
the centre. Converting those, then recomputing them per tile after cutting, is the step where a
silent error poisons everything downstream, because a mis-mapped label trains the detector on the
wrong targets and still reports plausible metrics. That is what the test suite exists to pin.

## Two splits, two sets of numbers

This repository holds results from two training runs over the same 649 tiles, differing only in how
pages were allocated to splits.

The **legacy** split assigns pages in lexicographic id order, which is arbitrary rather than
stratified. The consequence was severe: validation came entirely from book 4, and books 6 and 27,
which together supply 20 of the 34 pages and most of the training data, appeared in neither
evaluation split. Early stopping was driven by a single book, and the headline test score was
measured mostly on that same book. That run is
`YOLO_training/Scribal_Glyph_Charspotting_training_with_YOLOv8m_legacy.ipynb`.

The **stratified** split allocates each book's pages proportionally, so all four books appear
in all three splits.

| | Legacy | Stratified |
| --- | --- | --- |
| Train | 27 pages, 398 tiles, 6,562 label rows | 20 pages, 387 tiles, 5,642 label rows |
| Validation | 3 pages, 119 tiles, 1,430 rows | 7 pages, 127 tiles, 2,026 rows |
| Test | 4 pages, 132 tiles, 1,615 rows | 7 pages, 135 tiles, 1,939 rows |
| Books in train | 4:4, 6:10, 27:10, 29:3 | 4:6, 6:6, 27:6, 29:2 |
| Books in validation | 4:3 only | 4:2, 6:2, 27:2, 29:1 |
| Books in test | 4:3, 29:1 | 4:2, 6:2, 27:2, 29:1 |
| Classes with evaluation instances | 29 | 31 |

Both splits are page-disjoint. Tiles overlap by 25%, so splitting at tile level would put the same
pixels on both sides of the split. `tests/test_data_and_splits.py` asserts the property directly.

Ultralytics discards exact duplicate label rows at load time, so the instance counts it evaluates
are slightly below the row counts above. For the stratified split it reported 2,020 validation
instances (6 duplicates removed) and 1,928 test instances (11 removed).

## Results

All figures come from the `best.pt` checkpoint of each run, measured with Ultralytics 8.4.155 on a
Tesla T4. The model is a COCO-pretrained YOLOv8m at 512 pixels, batch 16, 25,871,026 parameters,
with horizontal flips disabled. Disabling flips matters here because glyphs are chiral: a mirrored
`b` is a `d`, so flip augmentation would teach the detector the wrong class.

Held-out test split:

| Run | Split | Training tiles | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- | --- |
| Task 2 | legacy | unmodified | 0.715 | 0.798 | 0.762 | 0.728 |
| Task 2 | stratified | unmodified | 0.587 | 0.730 | 0.711 | 0.658 |
| Task 3 | legacy | non-labelled regions blanked | 0.389 | 0.792 | 0.450 | 0.362 |
| Task 3 | stratified | non-labelled regions blanked | 0.344 | 0.723 | 0.418 | 0.341 |

Validation, same checkpoints:

| Run | Split | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| Task 2 | legacy | 0.750 | 0.785 | 0.806 | 0.765 |
| Task 2 | stratified | 0.568 | 0.691 | 0.672 | 0.622 |
| Task 3 | legacy | 0.350 | 0.861 | 0.516 | 0.411 |
| Task 3 | stratified | 0.341 | 0.635 | 0.386 | 0.319 |

### What correcting the split cost

The stratified baseline scores 0.711 mAP50 on test against the legacy split's 0.762, so the headline
falls by 5.1 points once every book appears in evaluation.

That comparison needs one adjustment before it means anything. mAP is an unweighted mean over the
classes that have instances in the evaluation split, and the two splits do not evaluate the same
classes. Legacy test contained 29; stratified test contains 31, adding `D` and `w`. Recomputing the
stratified means over the 29-class intersection gives mAP50 0.726 and mAP50-95 0.672. So roughly 1.5
of the 5.1 points is the class set changing rather than the split, and the split itself costs about
3.6 points of mAP50 and 5.6 of mAP50-95.

Precision absorbed most of the damage, falling from 0.715 to 0.587 while recall fell from 0.798 to
0.730. The legacy split was flattering the classifier more than the localiser. That is the pattern
you would expect if pages from an already-seen book were easy to classify because they share a
typeface, ink, press and scanning session with the training pages.

A 3.6-point correction is smaller than the severity of the split flaw might suggest. The honest
reading is that within-book evaluation inflated the result, by an amount worth reporting but not by
enough to overturn the baseline.

### What the task 2 and task 3 comparison shows

Blanking everything except the labelled characters costs 29.3 mAP50 points on the stratified test
split, 0.711 down to 0.418. On the 29-class intersection the gap is 29.6 points. The legacy run's
gap was 31.2 points on the same basis.

The two runs agree closely, and they agree on the shape of the failure as well as its size:

| | Legacy | Stratified |
| --- | --- | --- |
| Precision, task 2 to task 3 | 0.715 to 0.389 | 0.587 to 0.344 |
| Recall, task 2 to task 3 | 0.798 to 0.792 | 0.730 to 0.723 |
| mAP50 gap, 29-class basis | 0.312 | 0.296 |

Recall moves by less than a point in both runs while precision collapses. The detector still finds
character-shaped objects at roughly the same rate without surrounding page context, and is much
worse at judging which of them are genuine.

Replication across two different splits is worth something, because the split flaw was the most
obvious threat to this comparison and the comparison survived it. It is not an independent
replication, though, and the page lists show why. Five of the seven stratified test pages (10, 19,
20, 29 and 30) were legacy training pages, and three of the four legacy test pages (4, 5 and 6) are
now stratified training pages. The two runs reshuffle one 34-page corpus rather than sampling a new
one, so their results are entangled and cannot be pooled or averaged. Each run is internally
page-disjoint, which is what makes each one valid on its own terms.

Nor does replication make the comparison a clean context ablation, and it is not reported as one. Only the training tiles were modified.
`run_augmentations.py` reads the training split alone, so validation and test tiles remained
ordinary page images in both runs. Task 3 was therefore trained on one distribution and evaluated
on another, and the measured gap combines the loss of surrounding context with a mismatch between
the training and evaluation images. Both runs share that design, so replication shows the effect is
robust to the split, not that the mismatch has been removed.

The defensible claim is narrower than "context is worth 30 mAP points": a detector trained without
surrounding page context degrades sharply when it meets real pages, and that degradation reproduces
across two page allocations.

### Early stopping, and why final-epoch numbers are not quoted

Both runs were configured for 200 epochs with `patience=50`, and all four stopped early.

| Run | Split | Best epoch | Epochs run | Best validation mAP50 | Final-epoch validation mAP50 |
| --- | --- | --- | --- | --- | --- |
| Task 2 | legacy | 97 | 147 | 0.806 | 0.761 |
| Task 2 | stratified | 38 | 88 | 0.671 | 0.586 |
| Task 3 | legacy | 99 | 149 | 0.516 | 0.481 |
| Task 3 | stratified | 36 | 86 | 0.385 | 0.357 |

The stratified runs peaked roughly two and a half times earlier than the legacy ones, at epoch 38
against 97 and epoch 36 against 99. With a validation set spanning four books instead of one, the
best validation score arrives sooner and then decays.

That final-epoch column is worth reading carefully, because it documents a mistake this project
already made. An earlier write-up of the legacy run reported mAP50 0.761 for task 2 and 0.481 for
task 3, and both figures were later retracted. They are the final-epoch validation rows of
`YOLO_training/exp_train_project_task_*/results.csv`, taken from a checkpoint 50 epochs past the
best one and from the validation split rather than test. The correct legacy test figures are 0.762
and 0.450. For task 2 that lands within a point of the retracted number by coincidence, which is
part of why the error survived as long as it did. For task 3 the retracted 0.481 overstated the test
result by about three points.

The same trap is open in the stratified run. Task 2 finishes at 0.586 validation mAP50 against a
best of 0.671, a gap of 8.5 points across the 50 patience epochs, so anyone reading the last row of
the new `results.csv` would understate the run by about the same margin.

## Task 4: label completeness

Task 4 inverts the manipulation. It blanks the labelled regions instead of preserving them, then
runs inference over those modified training tiles. The question is about the ground truth rather
than about context: the annotations are sparse, so does the detector surface character instances
that were never labelled?

Inference used task 3's `best.pt` rather than the baseline detector, which is worth knowing when
reading the output, because task 3's model is the weaker of the two. The stratified run wrote
predictions for all 387 blanked training tiles.

No ground truth exists for the instances in question, so the stage is qualitative by construction.
No numerical result is reported for task 4 and none should be quoted.

## Limitations

- The stratified test split holds 1,928 instances across 31 classes, and the tail is thin. Eight
  classes have five or fewer test instances (`f`, `D`, `P`, `N`, `I`, `A`, `Q`, `w`), together 21
  instances, about 1.1% of the data, yet they carry 25.8% of the headline because mAP is a mean over
  classes. Their scores are close to bimodal: `D`, `P`, `N` and `Q` sit near 0.995 while `f` scores
  0.012 and `w` scores 0.005. Excluding all eight, the 23 remaining classes average 0.744 mAP50
  against a headline of 0.711.
- Class `f` fails in both runs, on one legacy test instance and on two stratified ones, so that
  failure is reproducible rather than a sampling artefact. Class `w` has no legacy test instances at
  all and scores 0.005 on its five stratified ones.
- Task 3's evaluation tiles were not blanked in either run, so its metrics conflate context loss
  with a train and evaluation mismatch. See the results section.
- Task 4 is qualitative by design. No metric exists for it.
- Under the stratified split, validation scores below test, 0.672 against 0.711. Validation and test
  now have identical book composition, two pages each from books 4, 6 and 27 plus one from book 29,
  so the remaining 4-point gap reflects page-level sampling rather than book identity. Under the
  legacy split the same comparison ran the other way, 0.806 against 0.762, because validation was
  three pages of a book the detector had already trained on.
- Thirty-four pages across four books is a small corpus, and seven test pages is a small evaluation
  set. Neither split changes that.
- There is no comparison against a human annotator and none against any prior method on this data.
  No claim is made about matching or replacing manual expert review.
- One architecture, one tile size, one overlap. None of them were ablated.
- The source annotations contain 47 exact duplicate rows, which Ultralytics silently discards at
  load time. The pipeline removes them where labels are written, which is why the label counts in
  the dataset table run slightly above the instance counts Ultralytics reports.

## Reproducing the stratified run

The stratified dataset is generated rather than committed, because it is derivable from the tiles
that are.

1. `python scripts/resplit_from_tiles.py` builds `data/dataset_stratified` from the committed
   `data/tiled_images` and `data/tiled_labels`. It reconstructs the page-to-book map, then verifies
   that map by re-deriving the legacy split and checking its book distribution against the table
   above before writing anything.
2. `python scripts/augment_stratified.py` writes the blanked training tiles for tasks 3 and 4. This
   stage reads the split tiles, not the original scans, so it runs without the source data.
3. Assemble `data/dataset_stratified_task3` as a copy of the stratified split with `images/train`
   replaced by the task 3 tiles, and `task4/images` alongside it.
4. Upload both dataset folders and run
   `YOLO_training/Scribal_Glyph_Charspotting_training_with_YOLOv8m_stratified.ipynb`. It detects Colab or
   Kaggle, derives every path from one root, refuses to reuse an existing run directory, and clears
   stale Ultralytics label caches before training. It stores no credentials.

5. `python scripts/detile_stratified.py` maps the tile predictions back to page coordinates and
   applies non-maximum suppression. `run_detiling_for_task` cannot be used for this run, because it
   builds its paths from the legacy layout, so the script passes explicit ones.

The legacy notebook is kept alongside the stratified one for provenance. It is the record of how the
superseded figures were produced, and it is not meant to be re-run: it points at the legacy dataset
paths and writes to run directories that already exist.

De-tiling needs the original page scans in `data/training-25plus/untiled_images` to recover true
page dimensions, and that set is not redistributed here. Download it from the competition link
above. Only the 34 pages from books {4, 6, 27, 29} belong there, in sorted filename order, because
page numbers index into that sorted list and a different file set would silently shift every page
assignment.

Page-level results, at the `conf=0.5` and `iou=0.45` operating point used for inference:

| Task | Split de-tiled | Pages | Detections |
| --- | --- | --- | --- |
| 2 | test | 7 | 1,735 |
| 3 | test | 7 | 5,033 |
| 4 | train | 20 | 12,154 |

Task 3 emits 5,033 page-level detections against 1,928 ground-truth test instances, roughly two and
a half times as many boxes as there are characters. That is the precision collapse in the results
table seen from a different angle: the context-stripped detector floods the page with candidates.

Two format notes for anyone comparing these files against the committed legacy artefacts under
`YOLO_training/results/detiled_predictions_task_*`. The new files carry six columns per row, adding
the confidence score that the older de-tiler did not write. And Ultralytics changed how it names
prediction files between the two runs, from a positional `image0.txt` to the source tile's own
`image_10_11.txt`. The de-tiler now resolves predictions by filename and falls back to the
positional name, so both conventions de-tile correctly. Matching by filename removes an ordering
assumption: under the positional scheme, a tile with no detections writes no file, which would
shift every subsequent index.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

64 tests pass, all on synthetic fixtures, so the suite runs on a fresh clone with no dataset present.
They cover the stages where a silent error corrupts every downstream result: tile geometry,
pseudo-YOLO parsing including class names containing spaces, the property that every label is
claimed by exactly one tile so characters are neither duplicated nor lost at tile seams, a full
round trip asserting that boxes land within one pixel of where they started, and page-disjointness
of the split.

## Not implemented

- Correlation analysis. Average the detected glyphs per class across a page or across pages from one
  book, then correlate that average template against the page. Peaks indicate matches, giving a
  classical alternative to YOLO where glyph shape varies little.
- Cross-book transfer. Repeat tiling, inference and de-tiling on Historia Scholastica pages to see
  whether the detector generalises to a book it was not trained on. The stratified split was the
  prerequisite, since every book now appears in evaluation and there is a baseline to compare a new
  book against.

## Requirements

See `requirements.txt`. Training needs a GPU; the pipeline and tests do not.
