# Microalgae identification

This repository contains the image processing code and research experiments for microalgae classification.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/microalgae/imaging/preprocessing/` | Image transformations shared by training and future inference. |
| `src/microalgae/imaging/features/` | Feature extractors and the LogGabor parameter file. |
| `training/classifiers/` | Classifier training, tuning, and evaluation. |
| `training/experiments/` | Research experiment entry points. |
| `configs/training.yaml` | Dataset, preprocessing, and experiment settings. |
| `data/` | Local datasets and generated feature tables. |
| `results/` | Generated experiment output. |
| `docs/references/` | Local research papers and reference documents. |
| `scripts/` | Dataset maintenance scripts. |
| `tests/` | Regression tests for feature extraction. |

`data/`, `results/`, and the reference documents are local research assets and are not packaged with `microalgae`.

## Running the existing experiment

From the repository root, install the project dependencies with the same Python interpreter you will use to run the experiment:

```powershell
py -m pip install -r requirements.txt
py main.py
```

`requirements.txt` covers image processing, feature extraction, classifier training, plotting, utility scripts, and tests. It includes the existing `phasepack==1.5` pin. `main.py` makes the local `src/microalgae` package importable directly. For development tools or other scripts that import `microalgae`, you can also install the package in editable mode with `py -m pip install -e .`.

Run the feature regression tests with `py -m pytest`. The experiment expects its input feature CSVs under `data/features/` as specified by `configs/training.yaml`.
