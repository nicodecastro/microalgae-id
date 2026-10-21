"""Inventory the configured dataset and audit an explicit feature selection."""

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from microalgae.imaging.dataset import build_dataset_manifest
from training.config import load_config
from training.feature_tables import (
    FEATURE_FILENAMES, audit_feature_tables, combine_feature_tables,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Training YAML (default: configs/training.yaml)")
    parser.add_argument("--features-dir", type=Path, help="Use separately regenerated feature tables")
    parser.add_argument("--combine", action="store_true", help="Export a keyed table only if all checks pass")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    dataset = config["dataset"]

    def project_path(value):
        path = Path(value)
        return path if path.is_absolute() else REPOSITORY_ROOT / path

    selected = config["features"]["feature_list"]
    if not selected or len(selected) != len(set(selected)) or set(selected) - FEATURE_FILENAMES.keys():
        parser.error("feature_list must select unique names from: " + ", ".join(FEATURE_FILENAMES))
    features_dir = project_path(args.features_dir or dataset["features_dir"])
    tables = {family: features_dir / FEATURE_FILENAMES[family] for family in selected}
    report_dir = project_path(config["experiment"]["results_dir"]) / "dataset_identity"
    manifest = build_dataset_manifest(project_path(dataset["raw_dir"]), dataset["classes"])
    issues = audit_feature_tables(manifest, tables)
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(report_dir / "dataset_manifest.csv", index=False)
    issues.to_csv(report_dir / "feature_table_audit.csv", index=False)
    print(f"Manifest: {len(manifest)} images; findings: {len(issues)}. Reports: {report_dir}")
    if not issues.empty:
        print(issues.groupby(["feature_block", "issue"], sort=False).size().to_string())
        return 1
    if args.combine:
        output = project_path(dataset["combined_features_dir"]) / "Combined_features_keyed.csv"
        combine_feature_tables(manifest, tables, output)
        print(f"Keyed feature table: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
