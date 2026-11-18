"""Fresh extraction, strict keyed alignment, and nested model evaluation."""

from collections.abc import Mapping
from datetime import datetime, timezone
import traceback

import yaml

from microalgae.imaging.dataset import build_dataset_manifest
from microalgae.imaging.preprocessing.configured import resolve_preprocessing
from training.evaluation import evaluate_nested_cv, prepare_evaluation, write_json
from training.splitting import load_parent_groups
from training.extraction import MODULES, _project_path, extract_configured_features
from training.feature_tables import FEATURE_FILENAMES, audit_feature_tables, combine_feature_tables


def run_configured_experiment(config, output_dir):
    """Run one fixed feature/preprocessing combination into a NEW directory.

    Reject unsafe paths and infeasible CV before creating output. Once a run
    starts, retain partial artifacts and status on failure; never resume or
    overwrite an existing run. Direct extractor defaults remain unchanged.
    """
    if not isinstance(config, Mapping):
        raise ValueError("Experiment configuration must be a mapping")
    root = _project_path(config["dataset"]["raw_dir"])
    output_dir = _project_path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be new: {output_dir}")
    if output_dir.is_relative_to(root):
        raise ValueError("Experiment output must not be inside the raw dataset")
    selected = config.get("features", {}).get("feature_list")
    if (not isinstance(selected, (list, tuple)) or not selected
            or any(not isinstance(name, str) or name not in MODULES for name in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("feature_list must select unique known feature families")
    preprocessing = {name: resolve_preprocessing(config, name) for name in selected}
    if not isinstance(config.get("logging", {}).get("save_intermediate_images", False), bool):
        raise ValueError("logging.save_intermediate_images must be a Boolean")
    manifest = build_dataset_manifest(root, config["dataset"]["classes"])
    missing_classes = set(config["dataset"]["classes"]) - set(manifest.Class)
    if missing_classes:
        raise ValueError(f"No images for configured classes: {sorted(missing_classes)}")
    mapping_path = config["dataset"].get("parent_groups_csv")
    if not isinstance(mapping_path, str) or not mapping_path.strip():
        raise ValueError("dataset.parent_groups_csv is required")
    manifest = load_parent_groups(manifest, _project_path(mapping_path))
    plan = prepare_evaluation(config.get("experiment", {}), manifest.Class.to_numpy(),
                              image_ids=manifest.image_id, parent_ids=manifest.parent_image_id)
    snapshot = yaml.safe_dump({
        "configuration": dict(config), "resolved_dataset_dir": str(root),
        "resolved_preprocessing": preprocessing, "standardize": False,
        "resolved_experiment": {**plan.settings, "param_grids": {
            name: grid for name, (_, grid) in plan.specifications.items()
        }},
    }, sort_keys=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    status = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat()}

    def progress(stage):
        status.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat())
        write_json(output_dir / "run_status.json", status)
        print(f"Experiment: {stage}", flush=True)

    try:
        progress("extraction")
        (output_dir / "resolved_settings.yaml").write_text(snapshot, encoding="utf-8")
        manifest.to_csv(output_dir / "dataset_manifest.csv", index=False)
        feature_dir = output_dir / "features"
        extract_configured_features(config, feature_dir, standardize=False)
        manifest.to_csv(feature_dir / "dataset_manifest.csv", index=False)
        progress("alignment")
        tables = {name: feature_dir / FEATURE_FILENAMES[name] for name in selected}
        # Use paths, not returned DataFrames, so sibling failure ledgers are checked.
        issues = audit_feature_tables(manifest, tables)
        issues.to_csv(output_dir / "feature_table_audit.csv", index=False)
        combined = combine_feature_tables(manifest, tables)
        combined.insert(len(combined.columns) - 2, "parent_image_id", manifest.parent_image_id.to_numpy())
        combined.to_csv(output_dir / "Combined_features_keyed.csv", index=False)
        progress("nested_cv")
        summaries = evaluate_nested_cv(combined, output_dir / "evaluation", plan, progress=progress)
        status["status"] = "completed"
        progress("completed")
        return {"output_dir": output_dir, "summary": summaries}
    except Exception as error:
        status.update(status="failed", error_type=type(error).__name__, reason=str(error),
                      traceback=traceback.format_exc(), updated_at=datetime.now(timezone.utc).isoformat())
        write_json(output_dir / "run_status.json", status)
        raise
