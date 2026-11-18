# Reviewed geometry crop pilot

This nine-image, reference-assisted pilot compares the saved full-image control
and H65 masks with the same two preparations applied after cropping. It does not
train classifiers, detect targets automatically, or change extraction defaults.

From the repository root, create a fresh proposal directory:

```powershell
.venv/Scripts/python.exe scripts/crop_geometry_pilot.py propose --output-dir results/geometry_diagnostic/crop_pilot_01
```

Open `review.md` in that directory. Each preview shows the original image and
proposed crop. Yellow marks the crop; cyan marks the selected existing reference.
The list deliberately includes tightly framed, low-resolution and contaminated
examples. There is no guarantee that every proposal is usable.

## Required human crop review

The only editable decision record is `crop_review.json`. For each image:

- Set `review_status` to `approved`, `no_useful_crop`, or `reference_insufficient`.
- Add observations and rejection reasons in `review_notes`.
- Adjust `box` when necessary: integer source-pixel coordinates
  `[left, top, right, bottom)`, with right and bottom exclusive.
- `target_add_polygon_indices` identifies existing additive reference polygons
  for the selected organism/colony. Preserve the whole colony; do not split its
  constituent cells. Existing subtractive polygons remain applied.

The assistant can record your decisions and show adjusted proposals. Approve
only the box you have reviewed. Approval confirms the selected reference is
adequate, the complete annotated target remains inside the box, and another
annotated target group is not included. A nearby structure of another species
or debris may still be present and must be assessed visually. The pilot cannot
infer its biological identity.

The proposed margins start at 10% of the selected target's larger bounding-box
dimension (minimum two original pixels), limited by image bounds. Boxes are
reference-assisted and derived from the coarse existing silhouette; they are
proposals, not ground-truth bounding boxes. Low-resolution images may not permit
a useful rectangle. Annotations, source identity fields and protected hashes
must remain unchanged. `other_annotated_target_pixels_in_box` and related fields
describe the original proposal; execution recalculates all checks after edits.

All nine dispositions must be resolved before evaluation. Do not approve all
boxes merely to unblock execution. Images marked unsuitable stay in coverage
reports without segmentation scores.

```powershell
.venv/Scripts/python.exe scripts/crop_geometry_pilot.py run --pilot-dir results/geometry_diagnostic/crop_pilot_01 --output-dir results/geometry_diagnostic/crop_pilot_01/evaluation_01
```

The evaluation directory must be new. Pending/invalid decisions fail before
output creation. The saved full-image predictions, their profile settings,
source-image identity and all 77 approved references are checked. Full-image
control/H65 predictions are reused from `comparison_10_hysteresis`; no historical
reference-reuse restriction is relaxed.

## Comparison protocol

Only approved original-resolution crops receive new segmentation. Each crop is
resized with aspect-preserving black padding to 256 x 256, then prepared with
the saved control or H65 settings. All parameters except their existing low
ratios (0.50/0.65) remain identical to the full-image experiment.

Predictions are stripped of crop padding, resized back into their original
source rectangle using nearest-neighbour interpolation, and mapped onto the
unchanged original 256 x 256 reference grid. Both full-image and crop predictions
are scored only inside the same rectangle. This tests scale and local
preprocessing effects without granting a free benefit for ignoring everything
outside a manually selected crop. Cropping changes the effective source-space
size of the fixed morphology kernels and minimum component-area threshold.
For an exact full-frame box, the original and crop preparation grids coincide;
mapping preserves the prediction directly rather than introducing a lossy
round-trip through a smaller source image. Other crop projections can introduce
nearest-neighbour discretization effects, which remain a limitation to report.

Foreground IoU, Dice, precision and recall retain the target-species reference
policy. Boundary F-scores use four-neighbour inner boundaries and Euclidean
many-to-one matches at 1/2/3 pixels on the common grid; 2 pixels is highlighted.
Artificial scoring-region borders are excluded before matching. Boundary
matching never crosses images. Within a species, pool overlap and boundary
counts before calculating metrics; then average species metrics equally.
Absent species are reported, not silently assigned zero or included as if
represented. No foreground/background binary mIoU or quality grade is reported.

## Outputs and completion

With all nine crops approved and processing successful, expect 18 new crop
predictions and 36 score rows. Outputs include lossless crops, approved boxes,
selected references, scoring regions, mapping metadata, common-grid predictions,
per-image/per-class/equal-species scores, both paired comparisons, coverage,
error ledgers, galleries and a Markdown findings report. The approved decision
manifest is frozen in the evaluation directory.

Empty predictions get zero overlap/F-score. A processing error remains an error:
it makes the run fail and removes that entire image from all four paired-score
conditions, preserving equal denominators and explicitly recording lost coverage.
Partial artifacts remain for diagnosis; retries require a fresh directory.

Complete the separate `visual_review.csv` after inspecting the galleries: use
`acceptable`, `reject`, or `uncertain`, with notes about target completeness,
contamination, boundaries and crop feasibility. The verification receipt starts
with visual review incomplete; it is not automatic human approval. The pilot is
not considered fully reviewed until those decisions have actually been made.

The unchanged 256 x 256 references limit fine contour conclusions. Numerical
ranking is exploratory and does not authorize crop-policy or H65 adoption.
The full-image baseline configurations, previous reports and annotations remain
unchanged.
