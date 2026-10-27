# Shared dataset feature export

The dataset feature export module consolidates traversal, image identity,
failure ledgers, optional dataset-wide standardization, and CSV writing for
all seven feature families. The existing `extract_features` functions remain
the caller interface. Each supplies a per-image adapter to the shared flow.

The adapter owns image loading, legacy or configured preprocessing, descriptor
calculation, previews, and family-specific exception context. Log-Gabor also
keeps one lazily constructed filter bank per export. An explicit `SkippedImage`
result lets geometry record a `ValueError`, warn, and continue without hiding
unexpected errors. Other families propagate failures as before.

This seam has seven real adapters. Export rules have locality in one module,
and export tests gain leverage across every family. The implementation does
not add a family registry, caching, concurrent extraction, or a second run
configuration interface.

## Compatibility guarantees

- Direct extractor arguments, return tables, and numerical calculations stay
  compatible, including configured preprocessing and preview paths.
- Geometry, Hu, LBP, and GLCM retain optional dataset-wide scaling. Nested CV
  still requests raw exports and fits its scaler within training folds.
- Feature names, metadata columns, variant filenames, and failure-ledger names
  retain their existing meanings.
- Geometry, Hu, and phase congruency sort filenames. The other families retain
  directory order. All families retain supplied class order.
- Geometry warns and records expected skips. Phase congruency rejects an empty
  class. The other families may pass an empty class if another class has rows.
- Every attempt resets its failure ledger. Image failures preserve an earlier
  feature table and block strict alignment. Successful retries clear failures.

## TDD evidence

Source: the architecture review's top recommendation, followed by the user's
request to implement one export flow for all seven feature families.

The user journey is exporting any supported feature family through the
existing direct or configured entry point, with reliable identity, numerical
results, and failure accounting.

| Guarantee | Evidence | Result |
| --- | --- | --- |
| Shared export contract exists | `python -m pytest tests/test_feature_export.py -q --tb=short` before implementation | RED: 10 tests executed and failed because `_export` was absent |
| Identity, feature-only scaling, order, skip accounting, retry, and empty selection/class policies | `python -m pytest tests/test_feature_export.py -q` after implementation | GREEN: 10 passed |
| Existing direct/configured behavior before refactoring | `python -m pytest tests/test_dataset_exports.py tests/test_configured_preprocessing.py -q` | 88 passed |
| Shared export and direct/configured behavior after routing all families | Same targets plus `tests/test_feature_export.py` | 98 passed |
| Every family's filename order, legacy descriptors, variants, previews, and error context | `python -m pytest tests/test_dataset_exports.py tests/test_lbp_features.py tests/test_glcm_features.py tests/test_efd_features.py tests/test_gabor_masked_statistics.py tests/test_phase_congruency_features.py -q` | 214 passed |
| Source compilation | `python -m compileall -q src/microalgae/imaging/features tests` | Passed |
| Full project regression suite, including configured extraction, nested CV, and desktop tests | `python -m pytest -q` | 409 passed, 27 dependency warnings in 109.91 seconds |

All commands used the repository's `.venv/Scripts/python.exe`.

The shared export contract tests executed all 51 executable source lines in
`_export.py`, measured with Python's standard-library tracing. Its executable
line inventory also included synthetic line 0; including that synthetic entry
produced 51/52 (98.1%). No coverage dependency was added.

No static type checker is configured or installed in the project environment.
Compilation and runtime checks were used; compilation is not static type
checking. Existing descriptor tests and manual-stage comparisons cover family
math; tests do not regenerate the real dataset or run a full-grid experiment.

The RED checkpoint is `74f9a0c`; the GREEN checkpoint is `6f5e633`. Keep these
results with the change if its checkpoint commits are later squashed.

## Review

The code-review skill reviewed standards and specification compliance in
separate agents. Both reported zero findings. The specification review also
compared direct-call interfaces and descriptor function syntax trees with
baseline `3db4f807e51fc9f58b97001eb605ece029e4d83a`.
