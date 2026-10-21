"""Configured extraction only: no classifier fitting or positional merging."""

from collections.abc import Mapping
from importlib import import_module
from pathlib import Path

import yaml

from microalgae.imaging.dataset import build_dataset_manifest
from microalgae.imaging.preprocessing.configured import resolve_preprocessing
from training.feature_tables import FEATURE_FILENAMES, audit_feature_tables


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "efd": "efd_features", "gabor": "gabor_features", "geom": "geometric_features",
    "glcm": "glcm_features", "hu": "hu_moments_features", "lbp": "lbp_features",
    "pc": "phase_congruency_features",
}


def _project_path(value):
    path = Path(value)
    return (path if path.is_absolute() else REPOSITORY_ROOT / path).resolve()


def extract_configured_features(config, output_dir):
    """Export selected families to a NEW directory and return their DataFrames.

    Accept a configuration dictionary so later experiments can reuse this path
    without editing YAML. Existing tables are never overwritten. Preflight
    errors write nothing; extraction errors retain per-image ledgers and an
    audit of all selected tables, then propagate to the caller.
    """
    if not isinstance(config, Mapping):
        raise ValueError("Extraction configuration must be a mapping")
    selected = config.get("features", {}).get("feature_list")
    if (not isinstance(selected, (list, tuple)) or not selected
            or any(not isinstance(name, str) or name not in MODULES for name in selected)
            or len(set(selected)) != len(selected)):
        raise ValueError("feature_list must select unique known feature families")
    settings = {name: resolve_preprocessing(config, name) for name in selected}
    save_images = config.get("logging", {}).get("save_intermediate_images", False)
    if not isinstance(save_images, bool):
        raise ValueError("logging.save_intermediate_images must be a Boolean")
    dataset = config["dataset"]
    root = _project_path(dataset["raw_dir"])
    output_dir = _project_path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output directory must be new: {output_dir}")
    if output_dir.is_relative_to(root):
        raise ValueError("Extraction output must not be inside the raw dataset")
    manifest = build_dataset_manifest(root, dataset["classes"])
    # Serialize before mkdir so unsupported configuration values cannot leave
    # a partial run directory during preflight.
    snapshot = yaml.safe_dump({
        "configuration": dict(config), "resolved_preprocessing": settings,
        "resolved_dataset_dir": str(root),
    }, sort_keys=False)
    extractors = {
        name: import_module(f"microalgae.imaging.features.{MODULES[name]}").extract_features
        for name in selected
    }
    tables = {name: output_dir / FEATURE_FILENAMES[name] for name in selected}
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest.to_csv(output_dir / "dataset_manifest.csv", index=False)
    (output_dir / "extraction_settings.yaml").write_text(snapshot, encoding="utf-8")
    frames = {}
    try:
        for name in selected:
            frames[name] = extractors[name](
                root, dataset["classes"], output_dir, save_images=save_images,
                preprocessing=settings[name],
            )
    finally:
        issues = audit_feature_tables(manifest, tables)
        issues.to_csv(output_dir / "feature_table_audit.csv", index=False)
    if not issues.empty:
        raise ValueError(f"Feature extraction audit failed; see {output_dir / 'feature_table_audit.csv'}")
    return frames
