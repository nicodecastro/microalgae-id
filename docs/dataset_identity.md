# Dataset identity and feature alignment

Every new dataset-level export contains its numeric feature columns followed
by `image_id` and `Class`. An ID is the image's root-relative path with forward
slashes, for example `n_palea/image_1.jpg`. Moving the entire dataset preserves
IDs. Renaming or moving individual images changes them. IDs are not file hashes;
replacing image contents at the same path does not change identity.

## Audit existing tables

From the repository root:

```powershell
.venv/Scripts/python.exe scripts/check_feature_tables.py
```

The command uses `configs/training.yaml` and the explicit seven-family filename
mapping in `training/feature_tables.py`. It writes `dataset_manifest.csv` and
`feature_table_audit.csv` under the configured results directory's
`dataset_identity/` folder. A failing audit exits with status 1. It never edits
feature tables or raw images. The manifest uses all six supported image
extensions (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`), configured class
order, and sorted filenames. Feature exports retain their previous row order.

The manifest's optional `parent_image_id` is blank until source relationships
are confirmed. Do not guess these links from filenames. They are not copied
into feature tables or used as features.

Legacy tables lack IDs and cannot be safely merged, even if row counts agree.
Do not backfill IDs or labels by row position. The initial local geometric
table also has 13 missing labels. Audit findings identify these problems;
regeneration, not positional repair, supplies fresh rows.

## Regenerate separately, then combine

Full extraction is a separate, deliberate run. Choose a fresh directory and
call the existing dataset extractors there, for example:

```python
from microalgae.imaging.features import lbp_features

frame = lbp_features.extract_features(
    "data/raw", ["m_aeruginosa", "n_fruticosa", "n_palea"],
    "data/features_keyed", standardize=False,
)
```

Repeat for the other selected families. Descriptor calculations, preprocessing,
scaling defaults, and variant filenames are unchanged. `standardize=False`
above is an explicit choice for raw LBP export, not a new default. Keep scaling
procedures consistent across any later model comparison.

Each attempt resets a sibling `<feature-file-stem>_failures.csv`, with columns
`image_id`, `Class`, `error_type`, and `reason`. A successful attempt leaves a
header-only ledger. Geometry still warns and skips unusable images; the other
extractors still raise. Failures are recorded immediately, including preview
failures. An unsuccessful retry does not replace an earlier feature table, and
its failure ledger prevents that stale table from passing strict alignment.
Do not remove failure reports to bypass validation; resolve the failures and
rerun extraction. A missing report is allowed for externally supplied keyed
tables, which still undergo all table checks.

```powershell
.venv/Scripts/python.exe scripts/check_feature_tables.py --features-dir data/features_keyed
.venv/Scripts/python.exe scripts/check_feature_tables.py --features-dir data/features_keyed --combine
```

Only an explicit `--combine` writes
`data/combined_features/Combined_features_keyed.csv` (or the configured combined
directory). All selected families must cover exactly the manifest IDs, with
unique nonblank IDs, matching nonblank labels, finite real numeric features,
and no recorded failures. Invalid inputs create no combined table and leave
any previous combined table untouched. Review the exit status, not just whether
an old output file exists.

The Python API accepts explicit mappings, including alternative GLCM/EFD
variants, rather than scanning every CSV in a directory:

```python
from microalgae.imaging.dataset import build_dataset_manifest
from training.feature_tables import audit_feature_tables, combine_feature_tables

manifest = build_dataset_manifest("data/raw", ["n_palea"])
tables = {
    "glcm": "data/features_keyed/GLCM_orientation_pooled_features.csv",
    "efd": "data/features_keyed/EFD_shape_normalized_features.csv",
}
issues = audit_feature_tables(manifest, tables)
combined = combine_feature_tables(manifest, tables)  # raises if issues exist
```

Combined rows follow manifest order. Numeric columns receive family prefixes
such as `geom__Area`; exactly one `image_id` and one `Class` follow them.
SVM/KNN exclude known identity metadata and validate numeric features/labels.
Valid legacy single-feature tables remain accepted by these readers. Training,
tuning, cross-validation, and scaling policies are otherwise unchanged.

This step prevents positional feature misalignment. It does not establish
source-group independence, scientific label correctness, or data-version
provenance. Parent grouping and full dataset regeneration remain separate work.
