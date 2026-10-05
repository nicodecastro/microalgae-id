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
| `src/microalgae/imaging/features/` | Feature extractors and the LogGabor parameter file. |
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
