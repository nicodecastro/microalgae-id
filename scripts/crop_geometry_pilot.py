"""Reviewed, reference-assisted crop pilot; never changes production preparation."""

import argparse
from copy import deepcopy
import json
from numbers import Integral
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

from microalgae.imaging.preprocessing.configured import prepare_feature_input, validate_preprocessing
from scripts._geometry_references import (
    ReferenceSession, child, digest, load_canvas, load_comparison, mask_path,
    rasterize, read_binary, validate_mask,
)
from scripts.diagnose_geometry import _project_path, _reference_canvas, _save_png, baseline_preprocessing
from scripts.score_geometry_masks import error_overlay, overlap_scores

COMPARISON = ROOT / "results/geometry_diagnostic/comparison_10_hysteresis"
REFERENCES = ROOT / "results/geometry_diagnostic/reference_02"
CONTROL = "otsu_background_k61_hysteresis_mask"
H65 = "otsu_background_k61_hysteresis_h65_mask"
PROFILES = (CONTROL, H65)
NAMES = {CONTROL: "Control", H65: "H65"}
SELECTION = (
    ("m_aeruginosa/img149_1_f8200.png", [5], "Central complete group; other groups and clipped organisms elsewhere."),
    ("m_aeruginosa/Acoi_img1_2_bc531.png", [4], "Upper group; very low resolution and adjacent annotated groups. A clean rectangle may be impossible."),
    ("n_fruticosa/3.jpg", [0], "Already tightly framed and low resolution; little cropping benefit expected."),
    ("n_fruticosa/image_3.jpg", [0], "Central colony; distant structure and scale bar can be excluded."),
    ("n_fruticosa/large (1).jpg", [0], "Whole colony; adjacent dense structure may remain inside the box. Contact is uncertain."),
    ("n_fruticosa/large.jpg", [0], "Central colony in a large field; nearby small structures may remain."),
    ("n_fruticosa/nfruticosa_or_nactinastroides3.jpg", [0, 1], "Whole central colony; surrounding small structures and background lines remain a concern."),
    ("n_palea/image_15.jpg", [0], "Whole annotated diatom; ruler markings cross it and cannot be removed by cropping."),
    ("n_palea/image_5.jpg", [0], "Whole annotated elongated target; faint ends and background specks need review."),
)
RESOLVED = {"approved", "no_useful_crop", "reference_insufficient"}
CROSS = generate_binary_structure(2, 1)
METRICS = ("iou", "dice", "precision", "recall")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def grid_transform(shape, canvas_shape=(256, 256)):
    """Match resize_with_padding's integer sizes and offsets, including rounding."""
    h, w = shape[:2]
    ch, cw = canvas_shape
    scale = min(cw / w, ch / h)
    nw, nh = int(w * scale), int(h * scale)
    if min(nw, nh) < 1:
        raise ValueError("Image aspect ratio produces an empty resized dimension")
    return {"source_size": [w, h], "canvas_size": [cw, ch],
            "resized_size": [nw, nh], "offset": [(cw - nw) // 2, (ch - nh) // 2]}


def to_canvas(mask, transform):
    nw, nh = transform["resized_size"]
    x, y = transform["offset"]
    cw, ch = transform["canvas_size"]
    result = np.zeros((ch, cw), bool)
    result[y:y + nh, x:x + nw] = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST) != 0
    return result


def from_canvas(mask, transform):
    nw, nh = transform["resized_size"]
    x, y = transform["offset"]
    return cv2.resize(mask[y:y + nh, x:x + nw].astype(np.uint8), tuple(transform["source_size"]),
                      interpolation=cv2.INTER_NEAREST) != 0


def validate_box(box, shape):
    h, w = shape[:2]
    if (not isinstance(box, list) or len(box) != 4
            or any(isinstance(v, bool) or not isinstance(v, Integral) for v in box)
            or not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)):
        raise ValueError("Crop box must be integer [left, top, right, bottom) within the source")
    return box


def map_crop(predicted, box, source_shape):
    left, top, right, bottom = validate_box(box, source_shape)
    if box == [0, 0, source_shape[1], source_shape[0]]:
        # Both preparation grids are identical. A trip through a smaller source
        # raster would otherwise invent differences for an unchanged full frame.
        content = to_canvas(np.ones(source_shape[:2], bool), grid_transform(source_shape))
        return predicted & content
    native = np.zeros(source_shape[:2], bool)
    transform = grid_transform((bottom - top, right - left))
    native[top:bottom, left:right] = from_canvas(predicted, transform)
    return to_canvas(native, grid_transform(source_shape))


def score_region(box, source_shape):
    left, top, right, bottom = validate_box(box, source_shape)
    region = np.zeros(source_shape[:2], bool)
    region[top:bottom, left:right] = True
    return to_canvas(region, grid_transform(source_shape))


def boundary_counts(predicted, reference, region):
    # Extract contours before restricting the scoring domain, avoiding crop-edge credit.
    interior = binary_erosion(region, structure=CROSS, border_value=0)
    pb = predicted & ~binary_erosion(predicted, structure=CROSS, border_value=0) & interior
    rb = reference & ~binary_erosion(reference, structure=CROSS, border_value=0) & interior
    npred, nref = int(pb.sum()), int(rb.sum())
    if not nref:
        raise ValueError("Selected reference has no evaluable boundary inside the scoring box")
    dp = distance_transform_edt(~rb)[pb]
    dr = distance_transform_edt(~pb)[rb] if npred else np.full(nref, np.inf)
    return {"predicted_boundary_pixels": npred, "reference_boundary_pixels": nref,
            **{f"matched_predicted_{t}px": int((dp <= t).sum()) for t in (1, 2, 3)},
            **{f"matched_reference_{t}px": int((dr <= t).sum()) for t in (1, 2, 3)}}


def boundary_metrics(counts):
    result = {}
    for t in (1, 2, 3):
        p = counts[f"matched_predicted_{t}px"] / counts["predicted_boundary_pixels"] if counts["predicted_boundary_pixels"] else 0.
        r = counts[f"matched_reference_{t}px"] / counts["reference_boundary_pixels"]
        result.update({f"boundary_precision_{t}px": p, f"boundary_recall_{t}px": r,
                       f"boundary_fscore_{t}px": 2 * p * r / (p + r) if p + r else 0.})
    return result


def selected_reference(annotation, indices):
    polygons = annotation["polygons"]
    if (not isinstance(indices, list) or not indices or len(indices) != len(set(indices))
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 or i >= len(polygons)
                   or polygons[i]["mode"] != "add" for i in indices)):
        raise ValueError("target_add_polygon_indices must select existing additive reference polygons")
    # Preserve original drawing order and every subtraction; never invent new masks.
    return rasterize([p for i, p in enumerate(polygons) if i in indices or p["mode"] == "subtract"],
                     tuple(annotation["canvas_shape"]))


def load_inputs(comparison_dir=COMPARISON, reference_dir=REFERENCES):
    directory, settings, selection, inventory, _ = load_comparison(comparison_dir)
    reference_dir = _project_path(reference_dir)
    annotation_data = json.loads((reference_dir / "annotations.json").read_text(encoding="utf-8"))
    session = ReferenceSession(annotation_data["comparison_dir"], reference_dir, resume=True)
    _, _, old_selection, old_inventory, _ = load_comparison(session.directory)
    if (inventory != old_inventory or baseline_preprocessing(settings) != baseline_preprocessing(session.settings)
            or not selection[["image_id", "Class"]].equals(old_selection[["image_id", "Class"]])):
        raise ValueError("Pilot source and references have incompatible provenance")
    if len(session.data["images"]) != 77 or any(r["status"] != "approved" for r in session.data["images"]):
        raise ValueError("Pilot requires unchanged 77 approved references")
    expected = validate_preprocessing({"foreground": {"enabled": True, "method": "otsu_background_hysteresis"}}, "geom")
    for profile, ratio in zip(PROFILES, (.5, .65)):
        expected["foreground"]["hysteresis"]["low_ratio"] = ratio
        if validate_preprocessing(settings["profiles"][profile], "geom") != expected:
            raise ValueError(f"Saved profile settings changed: {profile}")
    annotations = {r["image_id"]: r for r in session.data["images"]}
    paths = [p for p in reference_dir.rglob("*") if p.is_file()]
    paths += [directory / name for name in ("resolved_settings.yaml", "source_images.json", "selection.csv")]
    paths += [session.directory / "resolved_settings.yaml", session.directory / "source_images.json"]
    for identifier, _, _ in SELECTION:
        if identifier not in annotations or identifier not in inventory:
            raise ValueError(f"Missing pilot image: {identifier}")
        paths.append(child(Path(settings["dataset_dir"]), identifier))
        paths += [child(directory / "previews" / p, identifier + "_image.png") for p in PROFILES]
    return {"directory": directory, "reference_dir": reference_dir, "settings": settings,
            "annotations": annotations, "inventory": inventory,
            "labels": selection.set_index("image_id").Class.to_dict(),
            "hashes": {str(p.resolve()): digest(p) for p in paths}}


def check_hashes(hashes):
    for name, expected in hashes.items():
        if digest(Path(name)) != expected:
            raise ValueError(f"Protected input changed: {name}")


def safe_output(path, inputs):
    path = _project_path(path)
    forbidden = (Path(inputs["settings"]["dataset_dir"]).resolve(), inputs["directory"].resolve(),
                 inputs["reference_dir"].resolve(), ROOT / "configs", ROOT / "src", ROOT / "scripts")
    if any(path.is_relative_to(p) for p in forbidden):
        raise ValueError("Pilot output must be outside raw data, references, comparisons and source/config directories")
    if path.exists():
        raise FileExistsError(f"Output directory must be new: {path}")
    return path


def image_data(inputs, identifier):
    annotation = inputs["annotations"][identifier]
    canvas, content = load_canvas(inputs["settings"], annotation)
    path = child(Path(inputs["settings"]["dataset_dir"]), identifier)
    raw = cv2.imread(str(path))
    reference = read_binary(mask_path(inputs["reference_dir"], identifier), content.shape)
    validate_mask(reference, content)
    if not np.array_equal(reference, rasterize(annotation["polygons"], content.shape)):
        raise ValueError(f"Reference differs from polygons: {identifier}")
    return raw, canvas, content, reference


def proposed_box(selected, source_shape):
    native = from_canvas(selected, grid_transform(source_shape))
    ys, xs = np.nonzero(native)
    if not len(xs):
        raise ValueError("Selected reference disappears on the source grid")
    margin = max(2, int(np.ceil(.1 * max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))))
    h, w = source_shape[:2]
    return [max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
            min(w, int(xs.max()) + 1 + margin), min(h, int(ys.max()) + 1 + margin)]


def save_box_preview(path, raw, selected, box, title):
    left, top, right, bottom = box
    outline = from_canvas(selected, grid_transform(raw.shape))
    display = raw.copy()
    contours, _ = cv2.findContours(outline.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(display, contours, -1, (255, 255, 0), max(1, min(raw.shape[:2]) // 400))
    cv2.rectangle(display, (left, top), (right - 1, bottom - 1), (0, 200, 255), max(1, min(raw.shape[:2]) // 250))
    fig = Figure(figsize=(12, 6)); FigureCanvasAgg(fig)
    for ax, frame, label in zip(fig.subplots(1, 2), (display, display[top:bottom, left:right]), ("Original + proposed box", "Proposed crop (no segmentation)")):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); ax.set_title(label); ax.axis("off")
    fig.suptitle(title + f"\nBox [left, top, right, bottom): {box}"
                 + "\nYellow = proposed crop; cyan = selected existing reference", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .86)); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120); fig.clear()


def propose(output_dir):
    inputs = load_inputs()
    output = safe_output(output_dir, inputs)
    records = []
    for identifier, indices, notes in SELECTION:
        raw, _, content, reference = image_data(inputs, identifier)
        selected = selected_reference(inputs["annotations"][identifier], indices)
        if not selected.any() or np.any(selected & ~reference):
            raise ValueError(f"Invalid selected reference: {identifier}")
        box = proposed_box(selected, raw.shape)
        region = score_region(box, raw.shape) & content
        records.append({"image_id": identifier, "Class": inputs["labels"][identifier],
                        "source_sha256": inputs["inventory"][identifier]["sha256"],
                        "source_size": [raw.shape[1], raw.shape[0]], "box": box,
                        "target_add_polygon_indices": indices, "review_status": "pending", "review_notes": "",
                        "proposal_notes": notes,
                        "other_annotated_target_pixels_in_box": int((reference & ~selected & region).sum()),
                        "selected_reference_pixels_outside_box": int((selected & ~region).sum()),
                        "selected_reference_touches_source_border": bool(np.any(selected & ~binary_erosion(content, structure=CROSS))),
                        "low_source_resolution": min(raw.shape[:2]) < 128,
                        "original_transform": grid_transform(raw.shape)})
    check_hashes(inputs["hashes"])
    output.mkdir(parents=True)
    manifest = {"version": 1, "comparison_dir": str(inputs["directory"]),
                "reference_dir": str(inputs["reference_dir"]), "profiles": list(PROFILES),
                "selection_is_reference_assisted": True, "images": records, "protected_inputs": inputs["hashes"]}
    write_json(output / "crop_review.json", manifest)
    lines = ["# Review nine proposed crop boxes", "", "No crop segmentation has run. Yellow = proposed box; cyan = selected existing reference.", "",
             "Review the complete organism/colony, retained contaminants, faint boundaries, clipping and whether another annotated group enters the box.", "",
             "Edit `crop_review.json` (or give the assistant your decisions). Each image needs `review_status`: `approved`, `no_useful_crop`, or `reference_insufficient`. Record reasons in `review_notes`. Coordinates are integer source-pixel [left, top, right, bottom), with exclusive right/bottom edges. You may adjust `box` or select another existing additive polygon using `target_add_polygon_indices`.", "",
             "Approval confirms all selected reference pixels remain inside the box and no other annotated target group enters it. Other-species/debris pixels are not automatically identified by this check. Do not approve uncertain target identity or an inadequate reference. Rejected cases stay in coverage reports.", ""]
    for row in records:
        raw, _, _, _ = image_data(inputs, row["image_id"])
        selected = selected_reference(inputs["annotations"][row["image_id"]], row["target_add_polygon_indices"])
        relative = Path("proposals") / (row["image_id"] + "_box.png")
        save_box_preview(output / relative, raw, selected, row["box"], row["image_id"])
        lines += [f"## {row['image_id']}", "", row["proposal_notes"], "",
                  f"Other annotated target pixels inside box: **{row['other_annotated_target_pixels_in_box']}**. Selected pixels outside: **{row['selected_reference_pixels_outside_box']}**.", "",
                  f"![Proposed crop](<{(output / relative).resolve().as_posix()}>)", ""]
    (output / "review.md").write_text("\n".join(lines), encoding="utf-8")
    check_hashes(inputs["hashes"])
    print(f"Nine proposals saved: {output / 'review.md'}\nAwaiting explicit crop dispositions; no segmentation run.")
    return manifest


def validate_review(manifest, inputs):
    if (manifest.get("version") != 1 or manifest.get("profiles") != list(PROFILES)
            or manifest.get("comparison_dir") != str(inputs["directory"])
            or manifest.get("reference_dir") != str(inputs["reference_dir"])
            or manifest.get("protected_inputs") != inputs["hashes"]):
        raise ValueError("Crop review provenance or protected inputs changed")
    records = manifest.get("images", [])
    if (len(records) != len(SELECTION) or len({r["image_id"] for r in records}) != len(records)
            or {r["image_id"] for r in records} != {r[0] for r in SELECTION}):
        raise ValueError("Crop review must contain exactly the nine unique pilot IDs")
    resolved = []
    for record in records:
        identifier = record["image_id"]
        if record.get("review_status") not in RESOLVED:
            raise ValueError(f"Unresolved review disposition: {identifier}")
        raw, canvas, content, reference = image_data(inputs, identifier)
        if (record["source_sha256"] != inputs["inventory"][identifier]["sha256"]
                or record["Class"] != inputs["labels"][identifier]
                or record["source_size"] != [raw.shape[1], raw.shape[0]]
                or record["original_transform"] != grid_transform(raw.shape)):
            raise ValueError(f"Source identity changed: {identifier}")
        validate_box(record["box"], raw.shape)
        selected = selected_reference(inputs["annotations"][identifier], record["target_add_polygon_indices"])
        if not selected.any() or np.any(selected & ~reference):
            raise ValueError(f"Invalid selected reference: {identifier}")
        region = score_region(record["box"], raw.shape) & content
        if record["review_status"] == "approved":
            if np.any(selected & ~region):
                raise ValueError(f"Crop cuts selected reference: {identifier}")
            if np.any(reference & ~selected & region):
                raise ValueError(f"Another annotated target enters crop: {identifier}")
            boundary_counts(reference, reference, region)
        resolved.append((record, raw, canvas, content, reference, selected, region))
    return resolved


def aggregate(frame):
    tp, fp, fn = (int(frame[k].sum()) for k in ("true_positive_pixels", "false_positive_pixels", "false_negative_pixels"))
    result = {"iou": tp / (tp + fp + fn), "dice": 2 * tp / (2 * tp + fp + fn),
              "precision": tp / (tp + fp) if tp + fp else 0., "recall": tp / (tp + fn),
              "images": int(frame.image_id.nunique())}
    counts = {k: int(frame[k].sum()) for k in frame if k.startswith("matched_") or k.endswith("boundary_pixels")}
    return {**result, **boundary_metrics(counts)}


def check_score_coverage(frame, records, errors):
    """Require complete paired conditions for every successfully processed image."""
    failed = {row["image_id"] for row in errors}
    eligible = {row["image_id"] for row in records if row["review_status"] == "approved"} - failed
    expected = {(identifier, profile, preparation) for identifier in eligible
                for profile in PROFILES for preparation in ("full_image", "crop")}
    keys = list(frame[["image_id", "profile", "input"]].itertuples(index=False, name=None))
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete or inconsistent paired score coverage")


def save_gallery(path, identifier, raw, box, canvas, selected, region, views):
    fig = Figure(figsize=(15, 10)); FigureCanvasAgg(fig)
    axes = fig.subplots(2, 3).ravel()
    left, top, right, bottom = box
    frames = [(raw[top:bottom, left:right], "Approved source crop"), (canvas, "Common reference grid + selected boundary")]
    for ax, (frame, label) in zip(axes[:2], frames):
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)); ax.set_title(label); ax.axis("off")
        if label.startswith("Common"):
            ax.contour(selected, levels=[.5], colors="cyan", linewidths=.7)
            ax.contour(region, levels=[.5], colors="yellow", linewidths=.5)
    for ax, (label, (predicted, scores)) in zip(axes[2:], views.items()):
        ax.imshow(cv2.cvtColor(error_overlay(canvas, predicted, selected, region), cv2.COLOR_BGR2RGB))
        ax.set_title(f"{label}\nIoU {scores['iou']:.3f} | Dice {scores['dice']:.3f}\nP {scores['precision']:.3f} | R {scores['recall']:.3f} | BF2 {scores['boundary_fscore_2px']:.3f}"); ax.axis("off")
    for ax in axes[2 + len(views):]:
        ax.axis("off"); ax.text(.5, .5, "Processing error: see errors.csv", ha="center")
    fig.suptitle(identifier + "\nSame scoring box for every condition; green=TP, magenta=FN, orange=FP")
    fig.tight_layout(rect=(0, 0, 1, .94)); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120); fig.clear()


def write_results(output, frame, dispositions, errors):
    frame.to_csv(output / "per_image_scores.csv", index=False)
    pd.DataFrame(dispositions).to_csv(output / "coverage.csv", index=False)
    pd.DataFrame(errors, columns=["image_id", "profile", "input", "error"]).to_csv(output / "errors.csv", index=False)
    groups = []
    for (profile, preparation, label), group in frame.groupby(["profile", "input", "Class"]):
        groups.append({"profile": profile, "input": preparation, "Class": label, **aggregate(group)})
    summary_columns = ["profile", "input", "Class", *METRICS, "images",
                       *(f"boundary_{m}_{t}px" for t in (1, 2, 3) for m in ("precision", "recall", "fscore"))]
    summary = pd.DataFrame(groups, columns=summary_columns)
    summary.to_csv(output / "summary_by_class.csv", index=False)
    metrics = [*METRICS, *(f"boundary_fscore_{t}px" for t in (1, 2, 3))]
    equal = []
    if not summary.empty:
        for (profile, preparation), group in summary.groupby(["profile", "input"]):
            equal.append({"profile": profile, "input": preparation, "classes": len(group),
                          "images": int(group.images.sum()), **{m: group[m].mean() for m in metrics}})
    pd.DataFrame(equal, columns=["profile", "input", "classes", "images", *metrics]).to_csv(output / "equal_species_summary.csv", index=False)
    pairs = []
    indexed = frame.set_index(["image_id", "profile", "input"])
    for identifier, group in frame.groupby("image_id"):
        for profile in PROFILES:
            if all((identifier, profile, preparation) in indexed.index for preparation in ("full_image", "crop")):
                before, after = (indexed.loc[identifier, profile, p] for p in ("full_image", "crop"))
                pairs.append({"image_id": identifier, "Class": before.Class, "profile": profile,
                              **{m + "_crop_minus_full": after[m] - before[m] for m in metrics}})
    pd.DataFrame(pairs, columns=["image_id", "Class", "profile", *(m + "_crop_minus_full" for m in metrics)]).to_csv(output / "paired_crop_changes.csv", index=False)
    profile_pairs = []
    for (identifier, preparation), group in frame.groupby(["image_id", "input"]):
        if set(group.profile) == set(PROFILES):
            c, h = (indexed.loc[identifier, p, preparation] for p in PROFILES)
            profile_pairs.append({"image_id": identifier, "Class": c.Class, "input": preparation,
                                  **{m + "_h65_minus_control": h[m] - c[m] for m in metrics}})
    pd.DataFrame(profile_pairs, columns=["image_id", "Class", "input", *(m + "_h65_minus_control" for m in metrics)]).to_csv(output / "paired_profile_changes.csv", index=False)
    approved = sum(r["review_status"] == "approved" for r in dispositions)
    failed = {r["image_id"] for r in errors}
    lines = ["# Reference-assisted crop pilot", "", f"Reviewed sources: {len(dispositions)}; approved crops: {approved}; fully scored images: {frame.image_id.nunique()}; images with processing errors: {len(failed)}; score rows: {len(frame)} (expected {approved * 4} for approved cases).", "",
             "Both full-image and cropped predictions are scored inside the SAME approved rectangle on the unchanged original 256 x 256 reference grid. Outside pixels are ignored for both. Cropping precedes resizing. Pixel counts are pooled within species; species summaries are averaged equally. Species absent after review are not assigned invented zero scores; coverage and class counts must accompany comparisons.", "",
             "| Profile | Input | Classes | Images | Foreground IoU | Dice | Foreground precision | Foreground recall | BF 1 px | **BF 2 px** | BF 3 px |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in equal:
        lines.append(f"| {NAMES[row['profile']]} | {row['input']} | {row['classes']} | {row['images']} | " + " | ".join(f"{row[m]:.6f}" for m in metrics) + " |")
    lines += ["", "## Species coverage", "",
              "| Species | Selected | Approved | Scored | Rejected by review | Processing errors |",
              "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for label in sorted({r["Class"] for r in dispositions}):
        group = [r for r in dispositions if r["Class"] == label]
        counts = [len(group), sum(r["review_status"] == "approved" for r in group),
                  sum(r["processing_status"] == "scored" for r in group),
                  sum(r["processing_status"] == "excluded_by_review" for r in group),
                  sum(r["processing_status"] == "error" for r in group)]
        lines.append(f"| {label} | " + " | ".join(map(str, counts)) + " |")
    lines += ["", "## Interpretation and required visual review", "",
              "This small, deliberately selected, reference-assisted sample does not test automatic localization, detection recall, classifier performance, statistical significance, or independent generalization. No profile or crop policy is adopted. Cropping changes organism scale and the source-space meaning of kernel 61 and the 25-pixel component cutoff. The old reference resolution limits contour conclusions even when crops reveal finer detail.", "",
              "Boundary extraction uses four-neighbour inner contours and Euclidean many-to-one matching within each image. Exclude a one-pixel scoring-region border before matching; pool matched boundary counts within species and average species F-scores. Tolerances 1/2/3 pixels refer to the COMMON original grid, not the enlarged crop. Empty prediction scores are zero; errors are not converted into successful scores. Images with any processing error are absent from every condition's paired summary and remain explicitly recorded in coverage/errors. A full-frame crop maps identically without an unnecessary source-raster round-trip; other crops use the documented nearest-neighbour projection, which can introduce discretization effects.", "",
              "Review each gallery for complete silhouettes (including faint ends), retained contaminants, ruler interference, boundary agreement, and whether the crop is useful. Add findings to visual_review.csv with review_status acceptable/reject/uncertain and notes; these are human decisions. Numerical improvement does not substitute for visual approval. Do not assign universal quality grades.", "",
              "[Per-image scores](per_image_scores.csv) | [Per-species summaries](summary_by_class.csv) | [Crop changes](paired_crop_changes.csv) | [Profile changes](paired_profile_changes.csv) | [Coverage](coverage.csv) | [Errors](errors.csv) | [Frozen manifest](approved_manifest.json) | [Verification](verification.json)", ""]
    for row in dispositions:
        identifier = row["image_id"]
        lines += [f"- `{identifier}`: {row['review_status']}; {row.get('review_notes', '')}"]
        if row["review_status"] == "approved":
            relative = Path("galleries") / (identifier + "_comparison.png")
            lines += [f"  [Review gallery](<{relative.as_posix()}>)"]
    (output / "findings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(pilot_dir, output_dir):
    pilot = _project_path(pilot_dir)
    manifest_path = pilot / "crop_review.json"
    manifest_hash = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    inputs = load_inputs()
    output = safe_output(output_dir, inputs)
    resolved = validate_review(manifest, inputs)
    # Preflight also checks every reused prediction before creating output.
    for record, _, _, content, _, _, _ in resolved:
        for profile in PROFILES:
            read_binary(child(inputs["directory"] / "previews" / profile, record["image_id"] + "_image.png"), content.shape)
    if digest(manifest_path) != manifest_hash:
        raise ValueError("Crop decisions changed during preflight")
    check_hashes(inputs["hashes"])
    output.mkdir(parents=True)
    write_json(output / "approved_manifest.json", deepcopy(manifest))
    status = {"status": "running", "review_manifest_sha256": manifest_hash}
    write_json(output / "run_status.json", status)
    rows, errors, transforms = [], [], []
    try:
        for record, raw, canvas, content, reference, selected, region in resolved:
            if record["review_status"] != "approved":
                continue
            identifier, box = record["image_id"], record["box"]
            left, top, right, bottom = box
            cropped = raw[top:bottom, left:right]
            _save_png(output / "crops" / (identifier + ".png"), cropped)
            save_box_preview(output / "approved_boxes" / (identifier + "_box.png"), raw, selected, box, identifier)
            _save_png(output / "references" / (identifier + "_selected.png"), selected.astype(np.uint8) * 255)
            _save_png(output / "regions" / (identifier + "_region.png"), region.astype(np.uint8) * 255)
            transforms.append({"image_id": identifier, "box": box, "original": grid_transform(raw.shape),
                               "crop": grid_transform(cropped.shape), "mapping": "nearest-neighbour crop canvas -> source rectangle -> original canvas"})
            views, image_rows = {}, []
            for profile in PROFILES:
                for preparation in ("full_image", "crop"):
                    try:
                        if preparation == "full_image":
                            predicted = read_binary(child(inputs["directory"] / "previews" / profile, identifier + "_image.png"), content.shape)
                        else:
                            prepared = prepare_feature_input(cropped, inputs["settings"]["profiles"][profile], "geom")
                            _save_png(output / "crop_predictions" / profile / (identifier + "_mask.png"), prepared.image)
                            predicted = map_crop(prepared.image != 0, box, raw.shape)
                        counts = boundary_counts(predicted, reference, region)
                        scores = {**overlap_scores(predicted, reference, region), **counts, **boundary_metrics(counts)}
                        image_rows.append({"image_id": identifier, "Class": record["Class"], "profile": profile,
                                           "input": preparation, "prediction_status": "empty" if not predicted[region].any() else "nonempty", **scores})
                        views[NAMES[profile] + " / " + preparation] = predicted, scores
                        _save_png(output / "common_grid_predictions" / preparation / profile / (identifier + "_mask.png"), predicted.astype(np.uint8) * 255)
                    except Exception as error:
                        errors.append({"image_id": identifier, "profile": profile, "input": preparation, "error": str(error)})
            if len(image_rows) == 4 and not any(e["image_id"] == identifier for e in errors):
                rows.extend(image_rows)
            save_gallery(output / "galleries" / (identifier + "_comparison.png"), identifier, raw, box, canvas, selected, region, views)
        columns = ["image_id", "Class", "profile", "input", "prediction_status", *METRICS,
                   "true_positive_pixels", "false_negative_pixels", "false_positive_pixels",
                   "predicted_boundary_pixels", "reference_boundary_pixels",
                   *(f"matched_{side}_{t}px" for side in ("predicted", "reference") for t in (1, 2, 3)),
                   *(f"boundary_{metric}_{t}px" for t in (1, 2, 3) for metric in ("precision", "recall", "fscore"))]
        frame = pd.DataFrame(rows, columns=columns)
        check_score_coverage(frame, manifest["images"], errors)
        if not frame.empty:
            numeric = frame.select_dtypes(include=[np.number])
            if not np.isfinite(numeric.to_numpy()).all() or frame.duplicated(["image_id", "profile", "input"]).any():
                raise ValueError("Invalid pilot score coverage or non-finite values")
            bounded = [*METRICS, *(f"boundary_{m}_{t}px" for t in (1, 2, 3) for m in ("precision", "recall", "fscore"))]
            if not ((frame[bounded] >= 0) & (frame[bounded] <= 1)).to_numpy().all():
                raise ValueError("Pilot metrics must be bounded by zero and one")
        dispositions = [{"image_id": r["image_id"], "Class": r["Class"], "review_status": r["review_status"],
                         "review_notes": r["review_notes"], "processing_status": "excluded_by_review" if r["review_status"] != "approved"
                         else "error" if any(e["image_id"] == r["image_id"] for e in errors) else "scored"} for r in manifest["images"]]
        write_json(output / "transforms.json", transforms)
        write_results(output, frame, dispositions, errors)
        pd.DataFrame([{"image_id": r["image_id"], "profile": p, "input": preparation,
                       "review_status": "unreviewed", "notes": ""}
                      for r in manifest["images"] if r["review_status"] == "approved"
                      for p in PROFILES for preparation in ("full_image", "crop")],
                     columns=["image_id", "profile", "input", "review_status", "notes"]).to_csv(output / "visual_review.csv", index=False)
        check_hashes(inputs["hashes"])
        if digest(manifest_path) != manifest_hash:
            raise ValueError("Crop decisions changed during scoring")
        expected = 4 * sum(r["review_status"] == "approved" for r in manifest["images"])
        write_json(output / "verification.json", {"protected_inputs_unchanged": True, "review_manifest_unchanged": True,
                   "scored_rows": len(frame), "expected_rows": expected, "processing_errors": len(errors),
                   "paired_coverage_verified": True,
                   "common_scoring_grid": [256, 256], "boundary_tolerances_px": [1, 2, 3],
                   "highlighted_boundary_tolerance_px": 2, "selection_is_reference_assisted": True,
                   "visual_review_complete": False, "adoption_approved": False})
        status.update(status="failed" if errors else "completed", score_rows=len(frame), processing_errors=len(errors))
        write_json(output / "run_status.json", status)
        if errors:
            raise RuntimeError("Pilot processing errors; inspect errors.csv and coverage.csv")
        print(f"Pilot measurements saved: {output}; visual assessment remains required.")
        return frame
    except Exception as error:
        status.update(status="failed", error=str(error))
        write_json(output / "run_status.json", status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="stage", required=True)
    proposal = commands.add_parser("propose", help="Create nine crop proposals; no segmentation")
    proposal.add_argument("--output-dir", required=True, type=Path)
    evaluation = commands.add_parser("run", help="Score explicitly reviewed crops against full-image predictions")
    evaluation.add_argument("--pilot-dir", required=True, type=Path)
    evaluation.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.stage == "propose":
            propose(args.output_dir)
        else:
            run(args.pilot_dir, args.output_dir)
    except Exception as error:
        print(f"Crop pilot failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
