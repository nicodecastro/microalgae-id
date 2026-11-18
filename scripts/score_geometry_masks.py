"""Score saved geometry inputs against approved, identity-checked reference masks."""

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from scripts._geometry_references import (
    ReferenceSession, child, digest, load_canvas, load_comparison, mask_path, read_binary, validate_mask,
)
from scripts._canny_masks import ALL_CANNY_PROFILES
from scripts.diagnose_geometry import (QUALITY_PROFILE_NAMES, QUALITY_TUNING_PROFILES, _project_path, _save_png,
                                       baseline_preprocessing, HYSTERESIS_CONTROL, HYSTERESIS_EXPERIMENT_PROFILES)


def overlap_scores(predicted, reference, content):
    if (predicted.dtype != bool or reference.dtype != bool or content.dtype != bool
            or predicted.shape != reference.shape or content.shape != reference.shape
            or not reference[content].any()):
        raise ValueError("Overlap requires aligned Boolean masks and nonempty reference foreground")
    predicted, reference = predicted[content], reference[content]
    intersection = np.count_nonzero(predicted & reference)
    union = np.count_nonzero(predicted | reference)
    return {"iou": float(intersection / union),
            "dice": float(2 * intersection / (predicted.sum() + reference.sum())),
            "precision": float(intersection / predicted.sum()) if predicted.any() else 0.0,
            "recall": float(intersection / reference.sum()),
            "true_positive_pixels": int(intersection),
            "false_negative_pixels": int(np.count_nonzero(reference & ~predicted)),
            "false_positive_pixels": int(np.count_nonzero(predicted & ~reference))}


def error_overlay(canvas, predicted, reference, content):
    """BGR image: green overlap, magenta missed organism, orange extra background."""
    overlay = canvas.copy()
    for region, color in ((predicted & reference, (0, 200, 0)),
                          (reference & ~predicted, (220, 0, 220)),
                          (predicted & ~reference, (0, 140, 255))):
        region = region & content
        overlay[region] = np.rint(0.35 * canvas[region] + 0.65 * np.array(color)).astype(np.uint8)
    return overlay


def save_error_comparison(path, identifier, canvas, reference, views):
    profiles = [name for name in ("otsu_mask", "otsu_background_k31_mask", "otsu_background_k61_mask",
                                  "otsu_background_k61_hysteresis_mask") if name in views]
    if any(name in views for name in QUALITY_PROFILE_NAMES):
        profiles = [name for name in ("otsu_background_k61_mask", *QUALITY_PROFILE_NAMES) if name in views]
    if any(name in views for name in QUALITY_TUNING_PROFILES):
        profiles = [name for name in ("otsu_background_k61_mask", QUALITY_PROFILE_NAMES[0],
                    "otsu_background_k61_b2_mask", QUALITY_PROFILE_NAMES[1],
                    "otsu_background_k61_c1_mask", "otsu_background_k61_c2_mask") if name in views]
    if any(name in views for name in ALL_CANNY_PROFILES):
        profiles = list(dict.fromkeys([*profiles, "otsu_background_k61_mask",
                                       *(name for name in ALL_CANNY_PROFILES if name in views)]))
    if set(views) == set(HYSTERESIS_EXPERIMENT_PROFILES):
        profiles = list(HYSTERESIS_EXPERIMENT_PROFILES)
    columns = 3 if len(profiles) > 3 else 2
    rows = (len(profiles) + 1 + columns - 1) // columns
    figure = Figure(figsize=(5 * columns, 5 * rows))
    FigureCanvasAgg(figure)
    axes = figure.subplots(rows, columns).flat
    for axis in figure.axes:
        axis.axis("off")
    original = next(axes)
    original.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    original.contour(reference, levels=[0.5], colors="cyan", linewidths=0.8)
    original.set_title("Original + reference boundary (cyan)")
    original.axis("off")
    for axis, profile in zip(axes, profiles):
        overlay, scores = views[profile]
        axis.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        title = profile.removeprefix("otsu_background_").removesuffix("_mask").replace("_", " ")
        axis.set_title(f"{title}\nIoU {scores['iou']:.3f}\nPrecision {scores['precision']:.3f} | Recall {scores['recall']:.3f}", fontsize=10)
        axis.axis("off")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(identifier + "\nGreen = overlap | Magenta = missed organism | Orange = included background", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=130)
    figure.clear()


def hysteresis_ranking(frame):
    """Average each class equally, independently of its image count."""
    metrics = ["iou", "dice", "precision", "recall"]
    if frame.duplicated(["image_id", "profile"]).any():
        raise ValueError("Duplicate image/profile scores")
    if not np.isfinite(frame[metrics].to_numpy()).all():
        raise ValueError("Nonfinite hysteresis metrics")
    expected_classes = set(frame.Class)
    expected_ids = set(frame.image_id)
    if len(expected_classes) != 3 or HYSTERESIS_CONTROL not in set(frame.profile):
        raise ValueError("Hysteresis ranking requires its control and all three classes")
    for _, group in frame.groupby("profile"):
        if set(group.Class) != expected_classes or set(group.image_id) != expected_ids:
            raise ValueError("Each profile must cover the same images and classes")
    grouped = frame.groupby(["profile", "Class"])
    classes = grouped[metrics].agg(["mean", "median"])
    classes.columns = ["_".join(column) for column in classes.columns]
    classes["scored_count"] = grouped.size()
    classes = classes.reset_index()
    means = [f"{metric}_mean" for metric in metrics]
    ranking = classes.groupby("profile")[means].mean().rename(columns={
        f"{metric}_mean": f"macro_{metric}" for metric in metrics})
    overall = frame.groupby("profile")[metrics].mean().rename(columns={
        metric: f"image_weighted_{metric}" for metric in metrics})
    ranking = ranking.join(overall)
    for metric in metrics:
        ranking[f"macro_{metric}_difference_vs_hysteresis"] = ranking[f"macro_{metric}"] - ranking.loc[HYSTERESIS_CONTROL, f"macro_{metric}"]
        control = classes.loc[classes.profile == HYSTERESIS_CONTROL].set_index("Class")[f"{metric}_mean"]
        classes[f"{metric}_mean_difference_vs_hysteresis"] = classes[f"{metric}_mean"] - classes.Class.map(control)
    ranking = ranking.reset_index().sort_values(["macro_iou", "macro_dice", "profile"], ascending=[False, False, True])
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return classes, ranking


def write_hysteresis_report(frame, output):
    classes, ranking = hysteresis_ranking(frame)
    classes.to_csv(output / "summary_by_class.csv", index=False)
    ranking.to_csv(output / "profile_ranking.csv", index=False)
    pairs = frame.loc[frame.profile != HYSTERESIS_CONTROL].sort_values(["profile", "iou_difference_vs_hysteresis"])
    pairs.to_csv(output / "paired_vs_hysteresis.csv", index=False)
    lines = ["# Five-profile hysteresis experiment", "",
        "Primary score: average the three per-class mean IoUs with equal class weight. One shared configuration is used for every image. Overall image-weighted means are secondary.", "",
        "| Profile | Equal-class IoU | Equal-class Dice | Equal-class precision | Equal-class recall | Image-weighted IoU | Change in equal-class IoU |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in ranking.itertuples():
        lines.append(f"| {row.profile} | {row.macro_iou:.6f} | {row.macro_dice:.6f} | {row.macro_precision:.6f} | {row.macro_recall:.6f} | {row.image_weighted_iou:.6f} | {row.macro_iou_difference_vs_hysteresis:+.6f} |")
    lines += ["", "## Per-class results", "",
        "| Profile | Class | Images | Mean IoU | Mean precision | Mean recall | IoU change vs control |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in classes.itertuples():
        lines.append(f"| {row.profile} | {row.Class} | {row.scored_count} | {row.iou_mean:.6f} | {row.precision_mean:.6f} | {row.recall_mean:.6f} | {row.iou_mean_difference_vs_hysteresis:+.6f} |")
    lines += ["", "## Settings and interpretation", "",
        "- Control: low threshold 0.50 times the per-image Otsu threshold T; no bilateral or CLAHE.",
        "- H65/H80: change only the low ratio to 0.65/0.80.",
        "- HB: add bilateral (diameter 5, sigma color 15, sigma space 3) before Gaussian blur; retain ratio 0.50.",
        "- HC: add CLAHE (clip 1.5, tile grid 4x4) before Gaussian blur; retain ratio 0.50.",
        "- T is recalculated on each candidate's corrected residual. Kernel 61, Gaussian 3x3, 8-connectivity, closing, hole filling and 25-pixel component filtering remain fixed.",
        "- Class-level negative differences identify regressions. Precision/recall changes show contamination-versus-coverage tradeoffs; no recall cutoff is imposed.",
        "- References include only identifiable instances of the labeled target species. Other species, debris, rulers, text and padding are excluded. These foreground methods do not recognize species.",
        "- This is a failure-enriched exploratory sample, not independent validation. Only five images represent n_fruticosa. Numerical leadership does not approve adoption; visual review is required.", "",
        "## Outputs", "",
        "- [Per-image scores](per_image_scores.csv)", "- [Per-class means and medians](summary_by_class.csv)",
        "- [Equal-class ranking](profile_ranking.csv)", "- [Overall image-weighted means and medians](score_summary.csv)",
        "- [Paired candidate differences](paired_vs_hysteresis.csv)", "- [Six-panel reference galleries](error_comparisons/)",
        "- [Per-profile error overlays](error_overlays/)", "",
        "Historical profiles and reports remain available but are excluded from this five-profile experiment. Extraction defaults, annotations and training settings are unchanged.", ""]
    (output / "findings.md").write_text("\n".join(lines), encoding="utf-8")


def score_geometry_masks(comparison_dir, reference_dir, output_dir, *, reuse_references=False):
    output = _project_path(output_dir)
    if output.exists():
        raise FileExistsError(f"Score output must be new: {output}")
    reference_comparison = comparison_dir
    if reuse_references:
        metadata = json.loads((_project_path(reference_dir) / "annotations.json").read_text(encoding="utf-8"))
        reference_comparison = metadata["comparison_dir"]
    session = ReferenceSession(reference_comparison, reference_dir, resume=True)
    directory, settings, selection, inventory, settings_hash = load_comparison(comparison_dir)
    if reuse_references:
        _, _, old_selection, old_inventory, _ = load_comparison(reference_comparison)
        columns = ["image_id", "Class"]
        if (baseline_preprocessing(settings) != baseline_preprocessing(session.settings)
                or inventory != old_inventory
                or not selection[columns].set_index("image_id").sort_index().equals(
                    old_selection[columns].set_index("image_id").sort_index())):
            raise ValueError("Reference reuse requires identical image IDs, labels, source hashes, canvas shapes and baseline preprocessing")
    forbidden = [Path(settings["dataset_dir"]).resolve(), directory, session.directory,
                 session.output, Path(session.settings["dataset_dir"]).resolve(),
                 Path(settings["source_run"]).resolve(), Path(session.settings["source_run"]).resolve()]
    if any(output.is_relative_to(path) for path in forbidden):
        raise ValueError("Score output must be outside raw data, source run, comparison, and references")
    measurements = pd.read_csv(directory / "measurements.csv", keep_default_na=False)
    if not {"image_id", "profile", "extraction_status"}.issubset(measurements):
        raise ValueError("Missing diagnostic measurements")
    if measurements.duplicated(["image_id", "profile"]).any():
        raise ValueError("Duplicate diagnostic measurement identity")
    measurements = measurements.set_index(["image_id", "profile"])
    rows = []
    exclusions = []
    visuals = []
    hysteresis_experiment = settings.get("experiment") == "hysteresis_refinement"
    if hysteresis_experiment and (len(session.data["images"]) != 77
            or any(source["status"] != "approved" for source in session.data["images"])):
        raise ValueError("Hysteresis experiment requires all 77 references approved")
    annotations_hash = digest(session.path)
    reference_hashes = ({path: digest(path) for path in session.output.rglob("*") if path.is_file()}
                        if hysteresis_experiment else {})
    for source in session.data["images"]:
        identifier = source["image_id"]
        if source["status"] != "approved":
            exclusions.append({"image_id": identifier, "status": source["status"]})
            continue
        canvas, content = load_canvas(settings, source)
        reference = read_binary(mask_path(session.output, identifier), content.shape)
        validate_mask(reference, content)
        views = {}
        for profile in settings["profiles"]:
            # Profile names also form directory names; forbid traversal in snapshots.
            if profile != Path(profile).name or "/" in profile or "\\" in profile or profile in (".", ".."):
                raise ValueError(f"Invalid profile name: {profile}")
            key = (identifier, profile)
            if key not in measurements.index or measurements.loc[key, "extraction_status"] not in ("measured", "no_accepted_contours"):
                raise ValueError(f"Missing or failed diagnostic measurement: {key}")
            path = child(directory / "previews" / profile, identifier + "_image.png")
            predicted = read_binary(path, content.shape)
            scores = overlap_scores(predicted, reference, content)
            views[profile] = (error_overlay(canvas, predicted, reference, content), scores)
            rows.append({"image_id": identifier, "profile": profile,
                         "measurement_type": "edge_input_overlap" if profile == "canny_baseline" else "filled_mask_overlap",
                         "descriptor_valid": measurements.loc[key, "extraction_status"] == "measured",
                         **scores})
        visuals.append((identifier, canvas, reference, views))
    if not rows:
        raise ValueError("No approved reference masks to score")
    frame = pd.DataFrame(rows)
    baseline = frame.loc[frame.profile == "otsu_mask"].set_index("image_id")
    k61 = frame.loc[frame.profile == "otsu_background_k61_mask"].set_index("image_id")
    for metric in ("iou", "dice", "precision", "recall"):
        if not baseline.empty:
            frame[f"{metric}_difference_vs_dark_otsu"] = frame[metric] - frame.image_id.map(baseline[metric])
        if not k61.empty:
            frame[f"{metric}_difference_vs_kernel61"] = frame[metric] - frame.image_id.map(k61[metric])
    if hysteresis_experiment:
        frame = frame.merge(selection[["image_id", "Class", "baseline_status"]], on="image_id", validate="many_to_one")
        control = frame.loc[frame.profile == HYSTERESIS_CONTROL].set_index("image_id")
        for metric in ("iou", "dice", "precision", "recall"):
            frame[f"{metric}_difference_vs_hysteresis"] = frame[metric] - frame.image_id.map(control[metric])
    metrics = [c for c in ("iou", "dice", "precision", "recall", "iou_difference_vs_dark_otsu", "dice_difference_vs_dark_otsu") if c in frame]
    summary = frame.groupby(["profile", "measurement_type"], sort=False)[metrics].agg(["mean", "median"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary["scored_count"] = frame.groupby("profile").size().reindex(summary.index.get_level_values("profile")).to_numpy()
    summary["excluded_count"] = len(exclusions)
    tuning = frame.loc[frame.profile.isin(QUALITY_TUNING_PROFILES)].copy()
    if not tuning.empty:
        tuning["previous_profile"] = tuning.profile.map(QUALITY_TUNING_PROFILES)
        paired_keys = list(zip(tuning.image_id, tuning.previous_profile))
        for metric in ("iou", "dice", "precision", "recall"):
            previous = frame.set_index(["image_id", "profile"])[metric].loc[paired_keys].to_numpy()
            tuning[f"{metric}_difference_vs_previous_setting"] = tuning[metric].to_numpy() - previous
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "per_image_scores.csv", index=False)
    summary.reset_index().to_csv(output / "score_summary.csv", index=False)
    pd.DataFrame(exclusions, columns=["image_id", "status"]).to_csv(output / "excluded_references.csv", index=False)
    review = frame.loc[frame.profile == "otsu_background_k61_mask"]
    if not review.empty:
        review.sort_values("iou_difference_vs_dark_otsu").to_csv(output / "review_order.csv", index=False)
    candidate = frame.loc[frame.profile == "otsu_background_k61_hysteresis_mask"]
    if not candidate.empty and not hysteresis_experiment:
        candidate.sort_values("iou_difference_vs_kernel61").to_csv(output / "hysteresis_review_order.csv", index=False)
    quality = frame.loc[frame.profile.isin(QUALITY_PROFILE_NAMES)]
    if not quality.empty:
        quality.sort_values(["profile", "iou_difference_vs_kernel61"]).to_csv(output / "quality_review_order.csv", index=False)
    if not tuning.empty:
        tuning.sort_values(["profile", "iou_difference_vs_previous_setting"]).to_csv(output / "quality_tuning_review_order.csv", index=False)
    if hysteresis_experiment:
        write_hysteresis_report(frame, output)
    canny = frame.loc[frame.profile.isin(ALL_CANNY_PROFILES)].copy()
    if not canny.empty:
        canny = canny.merge(selection[["image_id", "Class", "baseline_status"]], on="image_id", validate="many_to_one")
        previous = frame.loc[frame.profile == "canny_50_100_mask"].set_index("image_id")
        if not previous.empty:
            for metric in ("iou", "dice", "precision", "recall"):
                canny[f"{metric}_difference_vs_canny50_100"] = canny[metric] - canny.image_id.map(previous[metric])
        canny.sort_values(["profile", "iou_difference_vs_kernel61", "image_id"]).to_csv(output / "canny_review_order.csv", index=False)
        ranking = canny.groupby("profile")[["iou", "dice"]].mean().reset_index().sort_values(
            ["iou", "dice", "profile"], ascending=[False, False, True])
        ranking.insert(0, "rank", range(1, len(ranking) + 1))
        ranking.to_csv(output / "canny_ranking.csv", index=False)
        best = ranking.iloc[0].profile
        reviews = pd.read_csv(directory / "review.csv", keep_default_na=False)
        chosen = reviews.loc[reviews.profile == best].merge(
            selection[["image_id", "Class", "baseline_status"]], on="image_id", validate="one_to_one")
        chosen.to_csv(output / "canny_selected_review.csv", index=False)
        chosen.groupby(["Class", "baseline_status", "review_status"]).size().rename("count").reset_index().to_csv(
            output / "canny_review_summary.csv", index=False)
        stratified = frame.loc[frame.profile.isin([*ALL_CANNY_PROFILES, "otsu_background_k61_mask"])].merge(
            selection[["image_id", "Class", "baseline_status"]], on="image_id", validate="many_to_one")
        stratified.groupby(["profile", "Class", "baseline_status"])[["iou", "dice", "precision", "recall"]].agg(
            ["mean", "count"]).to_csv(output / "canny_score_strata.csv")
        if (len(chosen) != len(selection) or set(chosen.image_id) != set(selection.image_id)
                or not chosen.review_status.isin(["unreviewed", "acceptable", "reject", "uncertain"]).all()):
            raise ValueError("Selected Canny review must cover every image with a valid review status")
        complete = chosen.review_status.isin(["acceptable", "reject", "uncertain"]).all()
        acceptable = int((chosen.review_status == "acceptable").sum())
        improved = bool(ranking.iloc[0].iou > k61.iou.mean())
        recommendation = "inconclusive; human review is pending"
        if complete:
            recommendation = ("unpromising under the exploratory gate" if not improved or acceptable <= 6
                              else "numeric gates passed; human assessment of recurring failure patterns is still required")
        (output / "canny_review_status.json").write_text(json.dumps({
            "selected_profile": best, "review_complete": bool(complete),
            "acceptable_count": acceptable,
            "reference_iou_improved": improved,
            "recurring_failure_pattern_assessment": "requires human assessment",
            "recommendation": recommendation,
            "review_source": str(directory / "review.csv"),
            "review_copy": "canny_selected_review.csv is a handoff snapshot; update source review.csv and rescore into a fresh directory",
            "adoption_approved": False,
        }, indent=2) + "\n", encoding="utf-8")
    for identifier, canvas, reference, views in visuals:
        for profile, (overlay, _) in views.items():
            _save_png(child(output / "error_overlays" / profile, identifier + "_overlay.png"), overlay)
        save_error_comparison(child(output / "error_comparisons", identifier + "_comparison.png"),
                              identifier, canvas, reference, views)
    if digest(session.path) != annotations_hash:
        raise ValueError("References changed during scoring; discard this report and use a fresh destination")
    if reference_hashes and ({p for p in session.output.rglob("*") if p.is_file()} != set(reference_hashes)
            or any(digest(path) != expected for path, expected in reference_hashes.items())):
        raise ValueError("Reference masks changed during scoring; use a fresh destination")
    (output / "scoring_settings.json").write_text(json.dumps({
        "comparison_dir": str(directory), "reference_dir": str(session.output),
        "reference_comparison_dir": str(session.directory), "reuse_references": reuse_references,
        "settings_sha256": settings_hash, "reference_settings_sha256": session.data["settings_sha256"],
        "annotations_sha256": digest(session.path),
        "selected_count": len(session.data["images"]), "scored_count": frame.image_id.nunique(),
        "excluded_count": len(exclusions), "evaluation_region": "real_image_content",
        "reference_set": "exploratory; not independent validation",
        "empty_prediction_precision": 0.0,
        "overlay_legend": {"green": "true positive", "magenta": "false negative", "orange": "false positive"},
    }, indent=2) + "\n", encoding="utf-8")
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reuse-references", action="store_true", help="Reuse references from their original comparison only after checking identical source images and baseline preprocessing")
    args = parser.parse_args(argv)
    try:
        scores = score_geometry_masks(args.comparison_dir, args.reference_dir, args.output_dir, reuse_references=args.reuse_references)
    except Exception as error:
        print(f"Geometry scoring failed: {error}", file=sys.stderr)
        return 1
    print(f"Scored {scores.image_id.nunique()} reference images. Scores do not approve segmentation or descriptors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
