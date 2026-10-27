# Domain glossary

## Feature family

A group of descriptors calculated with one method: geometry, Hu moments, LBP,
GLCM, EFD, Log-Gabor, or phase congruency. A family can have explicit variants,
such as orientation-pooled GLCM or shape-normalized EFD.

## Dataset feature export

The dataset-level operation that visits images for a feature family, records
failures, attaches image identity and class labels, optionally standardizes
numeric descriptors, and writes a feature table. Family calculations retain
their own preprocessing, descriptor parameters, and previews.

## Image ID

An image's dataset-root-relative path with forward slashes. It identifies a
path, not image contents or a verified independent specimen.

## Feature table

A table of numeric descriptors followed by `image_id` and `Class`. Tables from
different feature families align by image ID rather than row position.

## Failure ledger

A sibling `<feature-table-stem>_failures.csv` containing `image_id`, `Class`,
`error_type`, and `reason`. Each export attempt resets its ledger. Recorded
failures block strict feature-table alignment, including failures from skipped
images or unsuccessful retries that leave an older feature table in place.
