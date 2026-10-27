# Configured nested-CV experiments

Run from the repository root with the project environment:

```powershell
.venv/Scripts/python.exe main.py --config configs/training.yaml --output-dir results/nested_baseline
```

The output directory must not exist or lie inside the raw dataset. Relative
dataset and output paths resolve from the repository root; an explicitly
provided config path resolves from the current directory. No overwrite or
resume mode is provided. Existing feature CSVs are not inputs to this command.

## Protocol

1. Resolve the selected feature families and their fixed preprocessing settings.
   Validate class counts, CV settings, and parameter grids before creating output.
2. Extract fresh keyed tables with dataset-wide standardization disabled. Audit
   every selected table and failure ledger, then combine strictly by image ID
   in manifest order. Any missing/skipped image aborts training.
3. Evaluate SVM and KNN using the same five stratified, shuffled outer folds.
   For each outer training portion, tune on the same five inner folds with
   macro-F1 selection. Every candidate is a `StandardScaler`/classifier pipeline:
   the scaler learns only from that candidate's training fold. The winning
   pipeline is refitted on the entire outer training portion and predicts only
   the held-out outer test portion. Seed 42 determines all splits.
4. Finish all outer evaluation before final fitting. Retune each model on all
   images using a new shared five-fold search, then refit its winning pipeline
   on all images. These final pipelines are deployment artifacts, not another
   evaluation. No separate 80/20 holdout, post-tuning CV, or test-set KNN sweep
   runs in this workflow.

Changing preprocessing or selecting the best feature combination using these
outer scores can bias the reported result; this runner does not perform that
additional selection.

## Dataset independence and grouping

The current baseline uses image-level stratified nested CV and treats input
images as independent. The dataset owner reports no known related images.
This is a working assumption: a provenance or duplicate audit has not established
independence. Unknown parent IDs are a limitation to report, not evidence of
leakage or a blocker for baseline experiments.

The extraction/training pipeline does not create crops or augmented training
samples. It preprocesses the supplied images and extracts their descriptors;
saved previews are not added as samples. Existing dataset relationships, if any,
remain a separate question.

Different cells belonging to the same species are not, by that fact alone, a
grouping problem. Duplicates, overlapping crops, or repeated photographs of the
same specimen can make train/test images dependent and inflate evaluation scores.
Different specimens from one slide or acquisition session may also share imaging
conditions; source-level separation matters when claiming generalization to new
slides or sessions. Image-level scores do not establish that stronger claim.

If relevant source relationships are confirmed, record verified group IDs and
implement group-aware splitting across outer CV, inner CV, and final tuning.
The current runner does not implement grouped CV. Do not infer parent links from
filenames or group images merely because they share a class. A duplicate check
can strengthen confidence but cannot establish specimen or session independence.
See [scikit-learn's grouped-CV guidance](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data).

## Settings and grids

`features.feature_list` selects one feature combination, in order. Preprocessing
is fixed throughout the run. The `experiment` mapping contains:

```yaml
experiment:
  results_dir: "results"  # Retained for compatibility; --output-dir is authoritative.
  models: ["svm", "knn"]
  outer_folds: 5
  inner_folds: 5
  random_state: 42
  scoring: "f1_macro"
  n_jobs: -1
  refit_full_data: true
```

Outer folds run serially. Only inner searches are parallelized by `n_jobs`.
Both final tuning searches use `inner_folds`. Setting `refit_full_data: false`
explicitly skips deployment pipelines, leaving nested evaluation unchanged.
The old ambiguous `cross_validation` setting is rejected; migrate it to the
explicit counts above. Unknown experiment settings are rejected to catch typos.

The shared definitions in `training/classifiers/` preserve the full Chong grids:

- SVM: `C = 10^-5 ... 10^4`, kernels `linear/poly/rbf/sigmoid`, degrees 2–5,
  gamma `auto/scale` (320 candidates, including redundant kernel parameters).
- KNN: odd neighbors 1–41, weights `uniform/distance`, algorithms
  `auto/ball_tree/kd_tree/brute`, `p = 1/2` (336 candidates).

The grids follow Chong's [SVM script](https://github.com/RoyWeiiiii/Scope_3_Microalgae_shape_texture_convolution_classification/blob/main/SVM_model.py)
and [KNN script](https://github.com/RoyWeiiiii/Scope_3_Microalgae_shape_texture_convolution_classification/blob/main/k-NN_model.py).
Nested evaluation, fold-local scaling, and macro-F1 selection are project
choices, not a claim of exact reproduction of their evaluation protocol.

An optional `param_grids` mapping **replaces** the named model's entire grid.
Use unprefixed estimator parameter names; unspecified parameters keep estimator
defaults. A model with no override still uses its full grid. For a small smoke
test, add:

```yaml
  param_grids:
    svm:
      C: [1]
      kernel: ["linear"]
    knn:
      n_neighbors: [1]
      algorithm: ["ball_tree"]
```

Each outer training portion must contain at least five images per class for
inner CV, in addition to outer-CV feasibility. Every KNN candidate must fit the
smallest inner training portion (including final tuning). Impossible folds or
neighbors are errors; the runner never drops rows, shrinks grids, or lowers
fold counts. With the default grids, 5×5 evaluation plus full-data tuning/refits
requires **19,692 estimator fits**. It can take a long time and `n_jobs: -1`
can use substantial memory. Set a smaller positive worker count if needed.

## Saved artifacts

```text
run/
  run_status.json                 # Latest stage, timestamps, failure details
  resolved_settings.yaml          # Original config and effective grids/settings
  dataset_manifest.csv
  feature_table_audit.csv
  Combined_features_keyed.csv     # Raw numeric columns + image_id + Class
  features/
    *_features.csv               # Selected keyed tables (original family names)
    *_failures.csv
    dataset_manifest.csv
    extraction_settings.yaml     # standardize: false
    feature_table_audit.csv
    previews/                    # Only when configured
  evaluation/
    splits.json                  # Shared outer/inner/final indices and image IDs
    metadata.json                # Feature order, class order, versions, caveats
    summary.json
    svm/                         # Same layout for knn/
      outer_1/ ... outer_5/
        search_results.csv
        selection.json           # Inner selection score and parameters
        evaluation.json          # Outer-test metrics separately named
      fold_metrics.csv
      oof_predictions.csv        # Every image exactly once, plus fold number
      classification_report.json
      confusion_matrix.csv
      confusion_matrix.png
      summary.json
      final/
        search_results.csv
        selection.json           # Full-data CV selection score, not test score
        pipeline.joblib
        metadata.json
```

`outer_fold_summary` reports the unweighted mean and population standard
deviation (`std_ddof0`) across outer folds for accuracy, balanced accuracy,
macro precision/recall/F1. `pooled_oof_metrics` separately scores all out-of-fold
predictions together; these are not necessarily equal to fold means. Plots are
saved without opening windows. Metadata excludes IDs/labels from model inputs.

Preflight errors write nothing. Once extraction starts, failure status and
partial artifacts remain for inspection; do not treat an incomplete directory
as a successful experiment. Use a new directory for the next attempt. Successful
tables, completed fold results, per-image failure ledgers, and audits remain
available when later steps fail.

Load only trusted `pipeline.joblib` files. Present numeric inputs in the exact
`feature_columns` order recorded in final metadata. The saved pipeline includes
its scaler; do not separately standardize inference features. Image preprocessing
and descriptor calculation remain external to that pipeline and must match the
run's settings. Library versions are recorded for reproducibility.

## Compatibility and scope

The extraction-only command and direct extractors retain their existing defaults.
`extract_configured_features(config, output_dir, standardize=False)` now allows
raw geometry, Hu, LBP, and GLCM exports; the default remains `True`. EFD,
Log-Gabor, and phase congruency remain unstandardized in either case. The
experiment always requests raw exports and scales all selected numeric columns
inside model pipelines. Histogram/shape normalization intrinsic to a descriptor
is not dataset-wide standardization and remains unchanged.

`run_configured_experiment(config, output_dir)` is the reusable Python entry point.
`run_experiment(config_path=None, *, output_dir)` is a thin compatibility wrapper;
its output directory is now explicit. Legacy interactive classifier functions
remain available but are not called by this runner. No real dataset regeneration,
full-grid search, GUI wiring, Optuna, feature selection, or calibration is part
of the implementation verification.
