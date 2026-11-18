"""Compare saved Canny geometry with experimental masks; never train or approve."""

import argparse
import hashlib
from copy import deepcopy
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
import yaml

from microalgae.imaging.dataset import FAILURE_COLUMNS, image_id
from microalgae.imaging.features.geometric_features import (
    FEATURE_NAMES, _contour_features, extract_geometric_features,
)
from microalgae.imaging.preprocessing.configured import (
    prepare_feature_input, save_prepared_input, validate_preprocessing,
)
from microalgae.imaging.preprocessing.resize import resize_with_padding
from microalgae.imaging.preprocessing.geometry_foreground import CLAHE_PARAMETERS
from scripts._canny_masks import (
    ALL_CANNY_PROFILES, CANNY_MASK_PROFILES, CANNY_REFINEMENT_PROFILES,
    CANNY_MASK_OPERATIONS, prepare_canny_mask,
)
from training.feature_tables import _read_table, audit_feature_tables


PROFILE_NAMES = ("canny_baseline", "otsu_mask", "otsu_bright_mask", "adaptive_mask",
                 "otsu_background_k31_mask", "otsu_background_k61_mask")
QUALITY_PROFILE_NAMES = ("otsu_background_k61_bilateral_mask", "otsu_background_k61_clahe_mask",
                         "otsu_background_k61_bilateral_clahe_mask")
QUALITY_TUNING_PROFILES = {
    "otsu_background_k61_b2_mask": "otsu_background_k61_bilateral_mask",
    "otsu_background_k61_c1_mask": "otsu_background_k61_clahe_mask",
    "otsu_background_k61_c2_mask": "otsu_background_k61_clahe_mask",
}
HYSTERESIS_CONTROL = "otsu_background_k61_hysteresis_mask"
HYSTERESIS_EXPERIMENT_PROFILES = (HYSTERESIS_CONTROL, *(
    f"otsu_background_k61_hysteresis_{name}_mask" for name in ("h65", "h80", "hb", "hc")))
MEASUREMENT_COLUMNS = [
    "image_id", "Class", "baseline_status", "profile", "extraction_status",
    "component_count", "contour_count", "accepted_contour_count",
    "contour_point_counts", "contour_areas", "accepted_contour_areas",
    "foreground_fraction", "flags", *FEATURE_NAMES,
]


def _project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def baseline_preprocessing(settings):
    """Resolve canvas provenance for new snapshots and historical profile sets."""
    value = settings.get("baseline_preprocessing", settings.get("profiles", {}).get("canny_baseline"))
    if value is None:
        raise ValueError("Comparison requires baseline_preprocessing or a historical canny_baseline profile")
    return validate_preprocessing(value, "geom")


def _load_selection(run_dir, *, include_hysteresis=False, include_quality=False, include_quality_tuning=False, include_canny_masks=False, include_canny_refinements=False, hysteresis_experiment=False):
    if hysteresis_experiment and any((include_hysteresis, include_quality, include_quality_tuning, include_canny_masks, include_canny_refinements)):
        raise ValueError("--hysteresis-experiment cannot be combined with profile-expansion flags")
    snapshot = yaml.safe_load((run_dir / "features" / "extraction_settings.yaml").read_text(encoding="utf-8"))
    root = Path(snapshot["resolved_dataset_dir"]).resolve()
    if not root.is_dir():
        raise ValueError(f"Saved dataset directory is unavailable: {root}")
    baseline = validate_preprocessing(snapshot["resolved_preprocessing"]["geom"], "geom")
    if baseline["foreground"]["enabled"] or not baseline["segment"]["enabled"] or baseline["segment"]["method"] != "canny":
        raise ValueError("The source run must contain a saved Canny geometry baseline")
    manifest = _read_table(run_dir / "dataset_manifest.csv")
    geometry = _read_table(run_dir / "features" / "Geometric_shape_features.csv")
    failures = _read_table(run_dir / "features" / "Geometric_shape_features_failures.csv")
    issues = audit_feature_tables(manifest, {"geom": geometry})
    unexpected = issues.loc[issues.issue != "missing_image"]
    if not unexpected.empty:
        raise ValueError("Invalid saved geometry inputs:\n" + unexpected.to_string(index=False))
    if not set(FAILURE_COLUMNS).issubset(failures):
        raise ValueError("Saved geometry failure ledger is malformed")
    if failures.image_id.duplicated().any() or any(
        not isinstance(value, str) or not value.strip()
        for column in FAILURE_COLUMNS for value in failures[column]
    ):
        raise ValueError("Saved failure rows must have unique IDs and nonblank fields")
    expected = manifest.set_index("image_id")["Class"]
    for identifier, label in failures[["image_id", "Class"]].itertuples(index=False, name=None):
        if identifier not in expected or expected[identifier] != label:
            raise ValueError(f"Failure ledger identity/label conflict: {identifier}")
    failed_ids = set(failures.image_id)
    missing_ids = set(issues.loc[issues.issue == "missing_image", "image_id"])
    if failed_ids != missing_ids:
        raise ValueError("Failure IDs must exactly match images missing from the saved geometry table")
    classes = snapshot["configuration"]["dataset"]["classes"]
    if (not isinstance(classes, list) or not classes
            or any(not isinstance(label, str) or not label.strip() for label in classes)
            or len(set(classes)) != len(classes) or set(manifest.Class) - set(classes)):
        raise ValueError("Saved classes do not agree with the manifest")
    rng = np.random.default_rng(42)
    selected_ids = set(failed_ids)
    for label in classes:
        pool = sorted(geometry.loc[geometry.Class == label, "image_id"])
        selected_ids.update(rng.choice(pool, size=min(5, len(pool)), replace=False).tolist())
    selected = manifest.loc[manifest.image_id.isin(selected_ids), ["image_id", "Class"]].copy()
    selected["baseline_status"] = np.where(selected.image_id.isin(failed_ids), "failed", "control")
    for identifier in selected.image_id:
        # Reject escaping or noncanonical identifiers before creating output.
        if image_id(root / identifier, root) != identifier:
            raise ValueError(f"Noncanonical image ID: {identifier}")
    profiles = {"canny_baseline": baseline}
    for method in ("otsu", "otsu_bright", "adaptive"):
        settings = deepcopy(baseline)
        settings["segment"]["enabled"] = False
        settings["threshold"]["enabled"] = False
        settings["foreground"] = {
            "enabled": True, "method": method, "min_component_area": 25,
            "adaptive": {"block_size": 31, "c": 3},
        }
        profiles[f"{method}_mask"] = validate_preprocessing(settings, "geom")
    for kernel in (31, 61):
        settings = deepcopy(profiles["otsu_mask"])
        settings["foreground"].update(method="otsu_background", background={"kernel_size": kernel})
        profiles[f"otsu_background_k{kernel}_mask"] = validate_preprocessing(settings, "geom")
    if include_hysteresis:
        settings = deepcopy(profiles["otsu_background_k61_mask"])
        settings["foreground"]["method"] = "otsu_background_hysteresis"
        profiles["otsu_background_k61_hysteresis_mask"] = validate_preprocessing(settings, "geom")
    if include_quality or include_quality_tuning:
        for name in QUALITY_PROFILE_NAMES:
            settings = deepcopy(profiles["otsu_background_k61_mask"])
            # Isolate both factors from optional filtering in the source snapshot.
            settings["filter"]["enabled"] = "bilateral" in name
            if settings["filter"]["enabled"]:
                settings["filter"]["bilateral"]["params"] = [5, 15, 3]
            if "clahe" in name:
                settings["foreground"]["method"] = "otsu_background_clahe"
            profiles[name] = validate_preprocessing(settings, "geom")
    if include_quality_tuning:
        for name, previous in QUALITY_TUNING_PROFILES.items():
            settings = deepcopy(profiles[previous])
            if "_b2_" in name:
                settings["filter"]["bilateral"]["params"] = [5, 25, 3]
            else:
                settings["foreground"]["method"] = "otsu_background_clahe_clip1" if "_c1_" in name else "otsu_background_clahe_grid2"
            profiles[name] = validate_preprocessing(settings, "geom")
    if include_canny_masks or include_canny_refinements:
        for name, thresholds in CANNY_MASK_PROFILES.items():
            profiles[name] = {"diagnostic_method": "canny_filled", "thresholds": thresholds,
                              **deepcopy(CANNY_MASK_OPERATIONS)}
    if include_canny_refinements:
        for name, parameters in CANNY_REFINEMENT_PROFILES.items():
            profiles[name] = {"diagnostic_method": "canny_filled", **deepcopy(CANNY_MASK_OPERATIONS),
                              **deepcopy(parameters), "closing_kernel": "3x3 " + parameters["closing_shape"]}
    if hysteresis_experiment:
        control = deepcopy(profiles["otsu_background_k61_mask"])
        control["filter"]["enabled"] = False
        control["foreground"]["method"] = "otsu_background_hysteresis"
        profiles = {}
        for name in HYSTERESIS_EXPERIMENT_PROFILES:
            settings = deepcopy(control)
            if name.endswith("_h65_mask"):
                settings["foreground"]["hysteresis"]["low_ratio"] = 0.65
            elif name.endswith("_h80_mask"):
                settings["foreground"]["hysteresis"]["low_ratio"] = 0.8
            elif name.endswith("_hb_mask"):
                settings["filter"].update(enabled=True, bilateral={"params": [5, 15, 3]})
            elif name.endswith("_hc_mask"):
                settings["foreground"]["hysteresis"]["clahe"]["enabled"] = True
            profiles[name] = validate_preprocessing(settings, "geom")
    return root, selected, profiles


def _reference_canvas(image, settings):
    resize = settings["resize"]
    if resize["enabled"]:
        pad = 0 if resize["padding_color"] == "black" else image.mean(axis=(0, 1))
        return resize_with_padding(image, tuple(resize["target_size"]), pad, return_content_mask=True)
    return image.copy(), np.ones(image.shape[:2], dtype=bool)


def _measure(prepared, content, canvas):
    values = extract_geometric_features(prepared.image)
    contours, _ = cv2.findContours(prepared.image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    accepted = [_contour_features(contour) is not None for contour in contours]
    areas = [cv2.contourArea(contour) for contour in contours]
    accepted_areas = [area for area, valid in zip(areas, accepted) if valid]
    ys, xs = np.nonzero(content)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    touches = any(np.any(
        (contour[:, 0, 0] <= x0) | (contour[:, 0, 0] >= x1)
        | (contour[:, 0, 1] <= y0) | (contour[:, 0, 1] >= y1)
    ) for contour in contours)
    flags = []
    if values is None:
        flags.append("no_accepted_contours")
    if touches:
        flags.append("touches_content_border")
    if any(area < 25 for area in accepted_areas):
        flags.append("tiny_accepted_contour")
    overlay = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    for contour, valid in zip(contours, accepted):
        cv2.drawContours(overlay, [contour], -1, (0, 210, 0) if valid else (255, 0, 0), 1)
    count, _ = cv2.connectedComponents((prepared.image != 0).astype(np.uint8), connectivity=8)
    return {
        "extraction_status": "measured" if values is not None else "no_accepted_contours",
        "component_count": count - 1, "contour_count": len(contours),
        "accepted_contour_count": sum(accepted),
        "contour_point_counts": json.dumps([len(contour) for contour in contours]),
        "contour_areas": json.dumps(areas), "accepted_contour_areas": json.dumps(accepted_areas),
        "foreground_fraction": float(np.count_nonzero(prepared.image[content]) / content.sum()),
        "flags": ";".join(flags),
        **dict(zip(FEATURE_NAMES, values if values is not None else [None] * len(FEATURE_NAMES))),
    }, overlay


def _save_png(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to save diagnostic image: {path}")


def _save_comparison(path, identifier, raw, canvas, views):
    figure = Figure(figsize=(3.5 * (len(views) + 1), 7))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, len(views) + 1)
    for ax in axes.flat:
        ax.axis("off")
    if raw is not None:
        axes[0, 0].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"Original {raw.shape[1]} x {raw.shape[0]}")
        axes[1, 0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title("Reference canvas")
    else:
        axes[0, 0].text(0.5, 0.5, "Unreadable image", ha="center")
    for column, profile in enumerate(views, 1):
        view = views.get(profile)
        axes[0, column].set_title(profile.replace("_", " ") + "\ninput", fontsize=9)
        if view and "error" not in view:
            axes[0, column].imshow(view["prepared"], cmap="gray", vmin=0, vmax=255)
            axes[1, column].imshow(view["overlay"])
            flags = view["flags"].replace(";", "\n") or "No diagnostic flags"
            axes[1, column].set_title(f"{view['accepted']} accepted contours\n{flags}", fontsize=8)
        else:
            axes[0, column].text(0.05, 0.5, (view or {}).get("error", "Unavailable")[:180], wrap=True, fontsize=8)
    figure.suptitle(identifier + "\nGreen = numerically accepted; red = rejected. Neither is a quality approval.", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)


def diagnose_geometry(run_dir, output_dir, *, include_hysteresis=False, include_quality=False, include_quality_tuning=False, include_canny_masks=False, include_canny_refinements=False, hysteresis_experiment=False):
    """Write a deterministic, unapproved comparison set without training."""
    run_dir, output_dir = _project_path(run_dir), _project_path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Diagnostic output must be new: {output_dir}")
    include_quality = include_quality or include_quality_tuning
    include_canny_masks = include_canny_masks or include_canny_refinements
    root, selected, profiles = _load_selection(run_dir, include_hysteresis=include_hysteresis, include_quality=include_quality,
                                              include_quality_tuning=include_quality_tuning, include_canny_masks=include_canny_masks,
                                              include_canny_refinements=include_canny_refinements, hysteresis_experiment=hysteresis_experiment)
    canvas_settings = (validate_preprocessing(yaml.safe_load((run_dir / "features/extraction_settings.yaml").read_text(encoding="utf-8"))[
        "resolved_preprocessing"]["geom"], "geom") if hysteresis_experiment else profiles["canny_baseline"])
    if include_quality and profiles["otsu_background_k61_mask"]["filter"]["enabled"]:
        raise ValueError("Quality experiment requires an unfiltered control in the saved source run")
    if output_dir.is_relative_to(root) or output_dir.is_relative_to(run_dir):
        raise ValueError("Diagnostic output must be outside the raw dataset and source run")
    snapshot = yaml.safe_dump({
        "source_run": str(run_dir), "dataset_dir": str(root), "seed": 42,
        "controls_per_class": 5, "profiles": profiles,
        "mask_operations": {"gaussian_kernel": [3, 3], "closing_kernel": "3x3 ellipse", "fill_holes": True},
        "review_required": True,
        **({"experiment": "hysteresis_refinement", "baseline_preprocessing": canvas_settings,
            "ranking_objective": "equal_class_mean_iou", "control_profile": HYSTERESIS_CONTROL}
           if hysteresis_experiment else {}),
        **({"canny_masks": CANNY_MASK_OPERATIONS} if include_canny_masks else {}),
        **({"canny_refinements": CANNY_REFINEMENT_PROFILES} if include_canny_refinements else {}),
        **({"hysteresis": {"high": "Otsu on residual", "low_ratio": 0.5, "connectivity": 8}}
           if include_hysteresis else {}),
        **({"quality_preprocessing": {"bilateral_params": [5, 15, 3], "clahe_clip_limit": 1.5,
                                      "clahe_tile_grid_size": [4, 4],
                                      "order": ["bilateral_if_enabled", "clahe_if_enabled", "gaussian_3x3", "background_k61", "otsu", "cleanup"]}}
           if include_quality else {}),
        **({"quality_tuning": {"B2": {"bilateral_params": [5, 25, 3]},
                               "C1": {"clahe_clip_limit": 1.0, "clahe_tile_grid_size": [4, 4]},
                               "C2": {"clahe_clip_limit": 1.5, "clahe_tile_grid_size": [2, 2]}},
            "clahe_method_parameters": {name: {"clip_limit": values[0], "tile_grid_size": list(values[1])}
                                        for name, values in CLAHE_PARAMETERS.items()}}
           if include_quality_tuning else {}),
    }, sort_keys=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    selected.to_csv(output_dir / "selection.csv", index=False)
    selected.assign(selected=False).to_csv(output_dir / "reference_selection.csv", index=False)
    (output_dir / "resolved_settings.yaml").write_text(snapshot, encoding="utf-8")
    rows, reviews, errors, sources = [], [], [], []

    def error_record(identifier, profile, error):
        errors.append({"image_id": identifier, "profile": profile,
                       "error_type": type(error).__name__, "reason": str(error)})

    for number, (identifier, label, status) in enumerate(selected.itertuples(index=False, name=None), 1):
        print(f"Geometry diagnostics: {number}/{len(selected)} {identifier}", flush=True)
        views = {}
        raw, canvas, content = None, None, None
        try:
            raw = cv2.imread(str(root / identifier))
            if raw is None:
                raise ValueError(f"Failed to load image: {root / identifier}")
            canvas, content = _reference_canvas(raw, canvas_settings)
            sources.append({"image_id": identifier,
                            "sha256": hashlib.sha256((root / identifier).read_bytes()).hexdigest(),
                            "canvas_shape": list(canvas.shape[:2])})
        except Exception as error:
            error_record(identifier, "image", error)
            raw = None
        for profile, settings in profiles.items():
            record = {"image_id": identifier, "Class": label, "baseline_status": status, "profile": profile}
            try:
                if raw is None:
                    raise ValueError("Image unavailable; see errors.csv")
                prepared = (prepare_canny_mask(canvas, content, settings["thresholds"],
                                               closing_shape=settings.get("closing_shape", "ellipse"))
                            if profile in ALL_CANNY_PROFILES else prepare_feature_input(raw, settings, "geom"))
                measurements, overlay = _measure(prepared, content, canvas)
                record.update(measurements)
                views[profile] = {"prepared": prepared.image, "overlay": overlay,
                                  "accepted": measurements["accepted_contour_count"], "flags": measurements["flags"]}
                save_prepared_input(prepared, output_dir, profile, identifier)
                overlay_path = (output_dir / "overlays" / profile / identifier)
                _save_png(overlay_path.with_name(overlay_path.name + "_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            except Exception as error:
                error_record(identifier, profile, error)
                record.update(extraction_status="error", flags="processing_error")
                for feature in FEATURE_NAMES:
                    record[feature] = None
                views[profile] = {"error": str(error)}
            rows.append(record)
            reviews.append({"image_id": identifier, "profile": profile, "flags": record["flags"],
                            "review_status": "unreviewed", "notes": ""})
        try:
            path = output_dir / "comparisons" / identifier
            _save_comparison(path.with_name(path.name + "_comparison.png"), identifier, raw, canvas, views)
        except Exception as error:
            error_record(identifier, "comparison", error)
    measurements = pd.DataFrame(rows, columns=MEASUREMENT_COLUMNS)
    (output_dir / "source_images.json").write_text(json.dumps(sources, indent=2) + "\n", encoding="utf-8")
    measurements.to_csv(output_dir / "measurements.csv", index=False)
    pd.DataFrame(reviews).to_csv(output_dir / "review.csv", index=False)
    pd.DataFrame(errors, columns=["image_id", "profile", "error_type", "reason"]).to_csv(output_dir / "errors.csv", index=False)
    summary = {
        "status": "completed_with_errors" if errors else "completed",
        "selected_images": len(selected), "recorded_failures": int((selected.baseline_status == "failed").sum()),
        "successful_controls": int((selected.baseline_status == "control").sum()),
        "review_rows": len(reviews), "all_reviews_unapproved": True, "processing_errors": len(errors),
        "numerical_results": measurements.groupby(["profile", "extraction_status"]).size().rename("count").reset_index().to_dict("records"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if errors:
        raise ValueError(f"Geometry diagnostics encountered {len(errors)} processing/output errors; see {output_dir / 'errors.csv'}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Saved Canny experiment containing geometry results and a failure ledger")
    parser.add_argument("--output-dir", type=Path, required=True, help="Fresh diagnostic directory outside raw data and the source run")
    parser.add_argument("--include-hysteresis", action="store_true", help="Add kernel-61 residual hysteresis with low=0.5*Otsu, high=Otsu, 8-connectivity")
    parser.add_argument("--include-quality", action="store_true", help="Add bilateral-only, CLAHE-only and combined kernel-61 Otsu profiles")
    parser.add_argument("--include-quality-tuning", action="store_true", help="Add B2, C1, C2 and retain the earlier quality profiles for paired comparisons")
    parser.add_argument("--include-canny-masks", action="store_true", help="Add three fixed content-only Canny filled-mask candidates")
    parser.add_argument("--include-canny-refinements", action="store_true", help="Retain original Canny masks and add 20/40 and 30/60 elliptical closing, plus 50/100 and 20/40 rectangular closing")
    parser.add_argument("--hysteresis-experiment", action="store_true", help="Run only current hysteresis, H65, H80, HB and HC")
    args = parser.parse_args(argv)
    try:
        summary = diagnose_geometry(args.run_dir, args.output_dir, include_hysteresis=args.include_hysteresis,
                                    include_quality=args.include_quality, include_quality_tuning=args.include_quality_tuning,
                                    include_canny_masks=args.include_canny_masks,
                                    include_canny_refinements=args.include_canny_refinements,
                                    hysteresis_experiment=args.hysteresis_experiment)
    except Exception as error:
        print(f"Geometry diagnostics failed: {error}", file=sys.stderr)
        return 1
    print(f"Diagnostic reports completed for {summary['selected_images']} images. All reviews remain unreviewed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
