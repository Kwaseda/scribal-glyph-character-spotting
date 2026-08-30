# Scribal glyph character spotting

Character-level object detection on 15th-century printed Latin pages, built around a YOLOv8m
detector with a tiling and de-tiling pipeline.

Data comes from the [ICDAR 2025 / FAU competition on glyph detection in 15th-century European
printed documents](https://lme.tf.fau.de/competitions/icdar-2025-competition-on-glyph-detection-in-15th-century-european-printed-documents),
the `training-25plus` set, restricted to books {4, 6, 27, 29} and to pages set in minuscule. These
are incunabula, early printed books, not handwritten manuscripts, so the glyph shapes repeat far
more consistently than scribal hands would.

## The problem

Page scans are far too large to feed to a detector directly, and the targets are individual
characters, which are tiny relative to the page. The pipeline cuts each page into 512x512 tiles
that overlap by 128 pixels, trains on those, then maps predictions back into full-page coordinates
with non-maximum suppression so results are addressed in page space rather than tile space.

The ground-truth labels arrive in a pseudo-YOLO format: `<class name> x0 y0 w h`, where the
coordinates are **pixels rather than normalised fractions** and are anchored at the **upper-left
corner rather than the centre**. Converting those correctly, then recomputing them per tile after
cutting, is the step where a silent error poisons everything downstream. A mis-mapped label trains
the detector on the wrong targets and still reports plausible metrics. That is what the test suite
exists to pin.

## Dataset

| | Pages | Tiles | Label rows |
| --- | --- | --- | --- |
| Train | 27 | 398 | 6,562 |
| Validation | 3 | 119 | 1,430 |
| Test | 4 | 132 | 1,615 |

The split is **page-disjoint**. Train holds pages 1 to 3 and 10 to 33, validation holds 7 to 9, and
test holds 4 to 6 and 34. No page contributes tiles to more than one split. This matters more than
it might look: tiles overlap by 25%, so splitting at tile level would put the same pixels on both
sides of the split and leak the evaluation set into training. `tests/test_data_and_splits.py`
asserts the property directly.

Pages are assigned to splits in lexicographic id order, so page 10 is placed before page 2. That is
arbitrary rather than stratified, and it is the ordering that produced these results, so
`make_splits` preserves it deliberately.

### How the books fall across the splits

The 34 pages come from four books, and page numbers follow filename order, so book membership is
recoverable. The result is lopsided, and it is the most important thing to know before reading the
metrics:

| Book | Pages | Train | Val | Test |
| --- | --- | --- | --- | --- |
| WdB_004 | 10 | 4 | 3 | 3 |
| WdB_006 | 10 | 10 | 0 | 0 |
| WdB_027 | 10 | 10 | 0 | 0 |
| WdB_029 | 4 | 3 | 0 | 1 |

Validation is drawn entirely from book 4. Test is three pages of book 4 plus one of book 29. Books 6
and 27, which together supply 20 of the 34 pages and most of the training data, appear in neither
evaluation split. Early stopping was therefore driven by a validation set from a single book, and
the headline test score is measured mostly on that same book.

Page-disjoint is still the right property to hold, but it is weaker here than it sounds. Pages from
one book share a typeface, ink, press and scanning session, so the four book 4 training pages and
the three book 4 test pages are not independent in the way pages from different books would be. The
reported numbers describe how well the detector reads *more pages of a book it has already seen*.
They do not measure transfer to an unseen book, which is what the unimplemented cross-book stage
below was meant to test.

A corrected split is implemented and tested but **not yet trained on**:

```bash
python scripts/run_pipeline.py --split stratified
```

It allocates each book's pages proportionally, giving 20 training pages, 7 validation and 7 test,
with all four books present in all three splits:

| Book | Train | Val | Test |
| --- | --- | --- | --- |
| WdB_004 | 6 | 2 | 2 |
| WdB_006 | 6 | 2 | 2 |
| WdB_027 | 6 | 2 | 2 |
| WdB_029 | 2 | 1 | 1 |

The default remains `--split legacy`, which reproduces the committed dataset byte for byte, because
every metric below was measured on it. Switching strategies changes the dataset, so the numbers in
this README describe the legacy split only and the model has to be retrained before anything can be
quoted for the stratified one.

On class counts: the COCO category list declares **54** classes, only **31** of them occur anywhere
in the tiled data, and **29** have instances in the validation and test splits. Ultralytics averages
mAP over the classes present in the evaluation split, so the reported metrics are means over 29
classes. The class map is a mixture of lowercase letters, uppercase letters, a ligature, and a few
placeholder entries.

## Results

Metrics below come from the `best.pt` checkpoint of each run, measured on the **held-out test
split**: 132 tiles, 1,611 instances, from four pages the detector never trained or early-stopped on.
These are the numbers the notebook printed and the checkpoint that produced every prediction
artefact in this repository.

| Run | Training tiles | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- | --- |
| Task 2 | unmodified | 0.715 | 0.798 | **0.762** | **0.728** |
| Task 3 | non-labelled regions blanked | 0.389 | 0.792 | **0.450** | **0.362** |

Validation figures for the same checkpoints, for reference:

| Run | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| Task 2 | 0.750 | 0.785 | 0.806 | 0.765 |
| Task 3 | 0.350 | 0.861 | 0.516 | 0.411 |

Both runs were configured for 200 epochs with `patience=50` and both stopped early. Task 2 reached
its best epoch at 97 and stopped at 147; task 3 peaked at 99 and stopped at 149. The final-epoch
numbers in `results.csv` are therefore from a checkpoint 50 epochs past the best one, and are lower
than the figures above.

Training used a COCO-pretrained YOLOv8m at 512 pixels, batch 16, with horizontal flips disabled
(`fliplr=0.0`). Disabling flips is not incidental: glyphs are chiral, so a mirrored `b` is a `d`,
and flip augmentation would teach the detector the wrong class.

### What the task 2 and task 3 comparison shows

Blanking everything except the labelled characters costs **31 mAP50 points on the test set**, 0.762
down to 0.450. Precision takes the damage, falling from 0.715 to 0.389, while recall is flat at
roughly 0.79. The detector still finds character-shaped objects at about the same rate, but it is
much worse at judging which of them are genuine.

That comparison needs one caveat stated plainly, because it changes what the number means.
**Only the training tiles were modified.** `run_augmentations.py` reads the training split alone and
writes 398 blanked tiles; the validation and test tiles were left as ordinary page images. So task 3
was trained on one distribution and evaluated on another, and the measured drop combines two
effects that this experiment cannot separate: the loss of surrounding page context, and a mismatch
between the training and evaluation images.

The training logs show the mismatch directly. Task 3 reaches a *lower* final training classification
loss than task 2 (0.171 against 0.265), because blanked tiles are an easier target, while its
validation classification loss is more than three times higher (2.859 against 0.878). A model
trained and evaluated on the same blanked distribution would not show that pattern.

The defensible reading is therefore narrower than "context is worth 31 mAP points": a detector
trained without surrounding page context degrades sharply when it meets real pages. That is still a
real and useful result. It is not a clean context ablation, and it is not quoted as one.

## Task 4: label completeness

Task 4 inverts the manipulation. It blanks the labelled regions instead of preserving them, then
runs inference over those modified training tiles. The question is about the ground truth rather
than about context: the annotations are sparse, so does the detector surface character instances
that were never labelled?

Inference for this stage used **task 3's** `best.pt`, not the baseline detector. That is worth
knowing when reading the output, because task 3's model is the weaker of the two.

No ground truth exists for the instances in question, so the stage is qualitative by construction.
**No numerical result is reported for task 4 and none should be quoted.** Page-level predictions are
in `YOLO_training/results/detiled_predictions_task_4`.

## Layout

```
src/scribal_char_spotting/   tiling, label parsing, augmentation, de-tiling
scripts/                     entry points for each pipeline stage
tests/                       synthetic test suite, needs no dataset
configs/                     YOLO dataset yaml and the 54-class map
YOLO_training/               training logs, curves, weights, predictions
```

Trained weights are included, at roughly 50 MB each:

```
YOLO_training/exp_train_project_task_2/weights/best.pt
YOLO_training/exp_train_project_task_3/weights/best.pt
```

`YOLO_training/saved_models/` is empty; the local save in the training script never worked, so
training moved to Colab and the weights live in the Ultralytics run directories instead.

## Running it

The package resolves paths from its own location, so it works from any checkout. Set `SCRIBAL_ROOT`
to override that when the data lives elsewhere, as it does on Colab.

```bash
pip install -e .
```

1. `python scripts/run_pipeline.py` reads the page scans and pseudo-YOLO annotations, cuts
   overlapping 512x512 tiles, recomputes label coordinates per tile, drops empty tiles, builds the
   page-disjoint split, and writes the YOLO manifests.
2. Train with `configs/scribal-glyph-charspotting.yaml`. Its `path` key points at the Colab dataset
   mount and needs editing before a local run. The yaml defaults to the task 2 baseline; the comment
   above the key gives the task 3 variant.
3. `python scripts/run_augmentations.py` builds the blanked training tiles for tasks 3 and 4.
4. `python scripts/run_task2_detiling.py`, and the task 3 and task 4 equivalents, map tile
   predictions back to page coordinates and apply non-maximum suppression.

Steps 1, 3 and 4 need the original page scans in `data/training-25plus/`, which is not redistributed
here. Download it from the competition link above.

The de-tiling stage matches predictions to tiles by position in the sorted manifest, which is how
Ultralytics names its output files. It now checks that the two counts agree and fails loudly if they
do not, because a mismatch would silently attach every detection to the wrong tile origin.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

56 tests, all on synthetic fixtures, so the suite runs on a fresh clone with no dataset present.
They cover the stages where a silent error corrupts every downstream result:

- tile geometry, including that each tile equals the padded page cropped at its own coordinate, and
  that the tiles cover every padded pixel
- pseudo-YOLO parsing, including class names that contain spaces, and unknown classes raising rather
  than being guessed at
- the claim property the overlap design rests on: every label is claimed by exactly one tile, so
  characters are neither duplicated nor lost at tile seams
- a full round trip, tiling a synthetic page and feeding its own labels back through de-tiling, then
  asserting the boxes land within one pixel of where they started
- page-disjointness of the split, and that every tile of a page stays in one split

## Not implemented

The brief includes two further stages that are not built here:

- Correlation analysis. Average the detected glyphs per class across a page or across pages from one
  book, then correlate that average template against the page. Peaks indicate matches, giving a
  classical alternative to YOLO where glyph shape varies little.
- Cross-book transfer. Repeat tiling, inference and de-tiling on Historia Scholastica pages to see
  whether the detector generalises to a book it was not trained on. The stratified split described
  above is the prerequisite: until every book appears in evaluation, there is no baseline to compare
  a new book against.

## Limitations

- The test set holds 1,611 instances across 29 classes, and the tail is thin. Nine classes have five
  or fewer test instances (`C`, `f`, `P`, `x`, `S`, `N`, `I`, `A`, `Q`), together only 24 instances,
  about 1.5% of the data. Because mAP is a mean over classes, those nine carry 31% of the headline
  number while resting on almost no evidence. Class `f` has a single test instance and scores zero.
  The five commonest classes account for half of all test instances. Read the aggregate figures with
  that imbalance in mind.
- Task 3's evaluation tiles were not blanked, so its metrics conflate context loss with a train and
  evaluation mismatch. See the results section.
- Task 4 is qualitative by design. No metric exists for it.
- There is no comparison against a human annotator and none against any prior method on this data.
  No claim is made about matching or replacing manual expert review.
- One architecture, one tile size, one overlap. None of them were ablated.
- The split is page-disjoint but small and unbalanced by book: four test pages from a 34-page
  corpus, three of them from the same book that also supplies the entire validation set, and two of
  the four books absent from evaluation altogether. All of that follows from lexicographic ordering
  rather than from design. Treat the headline figure as within-book performance. The stratified
  split fixes the coverage gap, but no model has been trained on it yet, so nothing in this README
  is measured under it.
- The source annotations contain 47 exact duplicate rows, which Ultralytics silently discards at
  load time. The pipeline now removes them where labels are written, which is why the label counts
  in the dataset table are slightly higher than the instance counts Ultralytics reports.
- The committed prediction artefacts under `YOLO_training/results/` were produced before the current
  de-tiling code and have five columns per row. The de-tiler now also writes the confidence score,
  so regenerating them would produce six.

## Requirements

See `requirements.txt`. Training needs a GPU; the pipeline and tests do not.
