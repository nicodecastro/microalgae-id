# Microalgae identification

This repository contains the image processing code and research experiments for microalgae classification.

## Offline desktop prototype

The Windows-first review prototype opens local PNG, JPEG, and single-frame TIFF images. It supports multiple-image navigation, zoom, pan, and a synthetic demo with sample detection boxes. Imported images are **not classified**; the demo results are clearly labeled. Images and selections stay in memory for the current session, and source files are never modified.

From the repository root:

```powershell
py -m pip install PySide6
py desktop.py
```

Add images with the button, `Ctrl+O`, or drag-and-drop. Use **Open demo** to inspect the example result cards and overlays. Drag the image to pan, use the mouse wheel or toolbar to zoom, and use **Fit image** (`Ctrl+0`) to reset the view. The app makes no network requests while running. Install `pytest` and `pytest-qt` to run the UI tests; both are also listed in the full `requirements.txt`.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/microalgae/imaging/preprocessing/` | Image transformations shared by training and future inference. |
| `src/microalgae/imaging/features/` | Feature extractors, native Log-Gabor bank, and legacy LogGabor parameters. |
| `src/microalgae/domain/` | Detection and image-result data structures. |
| `src/microalgae/ui/` | PySide6 desktop prototype. |
| `training/classifiers/` | Classifier training, tuning, and evaluation. |
| `training/experiments/` | Research experiment entry points. |
| `configs/training.yaml` | Dataset, preprocessing, and experiment settings. |
| `data/` | Local datasets and generated feature tables. |
| `results/` | Generated experiment output. |
| `docs/references/` | Local research papers and reference documents. |
| `scripts/` | Dataset maintenance scripts. |
| `tests/` | Feature-extraction regression and desktop UI tests. |

`data/`, `results/`, and the reference documents are local research assets and are not packaged with `microalgae`.

## Running the existing experiment

From the repository root, install the project dependencies with the same Python interpreter you will use to run the experiment:

```powershell
py -m pip install -r requirements.txt
py main.py
```

`requirements.txt` covers image processing, feature extraction, classifier training, plotting, utility scripts, and tests. It includes the existing `phasepack==1.5` pin. `main.py` makes the local `src/microalgae` package importable directly. For development tools or other scripts that import `microalgae`, you can also install the package in editable mode with `py -m pip install -e .`.

Run the regression and UI tests with `py -m pytest`. The experiment expects its input feature CSVs under `data/features/` as specified by `configs/training.yaml`.

## Dataset identity and keyed feature alignment

New feature exports include `image_id` before `Class`. Audit tables with
`py scripts/check_feature_tables.py`; explicitly add `--combine` only after
separately regenerating keyed tables. Existing ID-less tables are not repaired
or merged by row order. See [the rollout guide](docs/dataset_identity.md) for
failure reports, fresh extraction, and strict combination.

## Configured feature extraction

Run `py scripts/extract_features.py --config configs/training.yaml --output-dir data/features_configured/baseline`
to extract selected features using shared preprocessing and per-family
overrides. The output directory must be new. No training runs automatically;
legacy direct extractor calls keep their original preprocessing. See the
[configured extraction guide](docs/configured_extraction.md) for defaults,
mask routing, previews, and experiment usage.

## Log-Gabor descriptors

Log-Gabor extraction now uses a full-resolution Python port of the MATLAB toolbox cited by Sanchez et al., with four scales and six orientations. It no longer requires the Python LogGabor package. The 964-column feature layout is unchanged, but the filter bank and corrected histogram mode/kurtosis change values: regenerate Gabor feature tables and retrain models when adopting it. See [the extraction notes](docs/log_gabor_features.md) for settings, mask rules, usage, upstream attribution, and reproduction limits.
