# Configured preprocessing and extraction

The extraction-only command connects preprocessing settings to the actual
descriptor inputs. It does not train models, combine tables, or overwrite a
previous run.

```powershell
.venv/Scripts/python.exe scripts/extract_features.py --config configs/training.yaml --output-dir data/features_configured/baseline
```

The output directory must not exist and must not be inside the raw dataset.
Relative dataset and output paths are resolved from the repository root.
Only families in `features.feature_list` are extracted, in the listed order.
All six supported image extensions remain included, matching the identity
manifest rather than the older, narrower `dataset.image_format` setting.

## Shared settings and family overrides

`configs/training.yaml` contains shared preprocessing settings and optional
`preprocessing.overrides.<family>` mappings. Overrides merge recursively;
an omitted value inherits its shared setting. Families are `geom`, `hu`, `lbp`,
`glcm`, `efd`, `gabor`, and `pc`. Unknown settings and family names are errors.
Grayscale is required for these descriptors.

The configured baseline preserves aspect ratio on a black 256 x 256 canvas.
It retains the existing `INTER_AREA` resize interpolation. `padding_color:
mean` instead uses the source image's mean (per channel before grayscale).
The canvas stores that mean in the source image's uint8 representation.

| Family | Baseline preparation | Descriptor input |
| --- | --- | --- |
| Geometry | Grayscale, Canny thresholds 150 and 175 | Edge image |
| Hu | Grayscale, bilateral (40, 1, 25), adaptive threshold block 11 and C 2 | Thresholded image |
| LBP / GLCM | Grayscale | Entire grayscale image, including padding |
| EFD | Grayscale, separate Sanchez segmentation with dilation 4 | Binary foreground mask |
| Log-Gabor | Grayscale, separate Sanchez segmentation with dilation 4 | Masked intensities and foreground mask |
| Phase congruency | Grayscale | Entire grayscale image, including padding |

These are reference-inspired starting settings, not verified optimal settings
or exact paper reproduction. Chong's reported best preprocessing differs by
family; the project's padded 256 resolution is a deliberate departure from
the reference's 512 setup. The existing Sanchez-style mask implementation,
filter settings, and descriptor conventions are not redesigned here.

Preparation order is resize, grayscale, bilateral filtering, adaptive
thresholding, and edge detection. Disabled stages do not run. EFD and Gabor
mask generation branches after grayscale/denoising and before thresholding or
edges. It does not resize again. EFD uses the mask's outline. Gabor retains
pre-filter masking and foreground-only statistics. Standalone GLCM and PC do
not acquire a mask.

Stage controls use these keys:

- `resize`: `enabled`, `target_size` (width, height), `padding_color` (`black` or `mean`).
- `filter`: `enabled`, `method: bilateral`, `bilateral.params` (diameter, sigma color, sigma space).
- `threshold`: `enabled`, `method: adaptive`, `adaptive.block_size` (odd, at least 3), `adaptive.c`.
- `segment`: `enabled`, `method` (`canny` or `sobel`), `canny.thresholds` (low, high), `sobel.ksize` (odd, 1 to 31).
- `mask`: `method: sanchez`, `dilation_size` (positive integer).

Parameters for disabled stages are retained but not evaluated. Explicit
threshold/edge experiments are supported for LBP, GLCM, Gabor, and PC. For
example, add this family override to compare Sobel-prepared LBP:

```yaml
preprocessing:
  overrides:
    lbp:
      segment:
        enabled: true
        method: sobel
        sobel:
          ksize: 3
```

Merge this fragment into the existing configuration rather than adding a
second YAML `preprocessing` key. EFD rejects enabled threshold/edge image
stages because it consumes its separate mask. Configured Gabor requires
enabled 256 x 256 resizing; incompatible sizes are rejected during preflight,
not silently corrected. Other families can override size or disable resizing.

## Outputs and failures

Each run contains the usual feature CSVs and failure ledgers, a
`dataset_manifest.csv`, `extraction_settings.yaml`, and
`feature_table_audit.csv`. The settings snapshot stores both the submitted
configuration and the resolved per-family preparation settings. Descriptor
modes and scaling defaults are unchanged: geometry, Hu, LBP, and GLCM retain
dataset-wide standardization by default; other exports remain unstandardized.
Python callers can pass keyword-only `standardize=False` for raw exports of
those four families. The snapshot records this flag. The extraction-only
command preserves the original `True` default; the
[nested experiment runner](nested_experiments.md) requests raw values and fits
scaling inside training folds.

With `logging.save_intermediate_images: true`, lossless previews are saved as
`previews/<family>/<image_id>_image.png`, plus `_mask.png` for EFD/Gabor.
These are the exact prepared calculation inputs, not a separately processed
preview. PC also retains its moment-map previews. Preview failures are recorded
as per-image failures, just like preprocessing and descriptor failures.

Geometry still warns and skips unusable images. Other extractor exceptions
stop the run. Partial outputs and failure ledgers remain available, and the
runner audits all selected table paths before propagating extraction errors.
Skipped/missing images or other audit findings also make the command fail.
Fix the cause and choose a new output directory for a retry.

After a successful run, use the existing explicit audit/combination command:

```powershell
.venv/Scripts/python.exe scripts/check_feature_tables.py --features-dir data/features_configured/baseline --combine
```

Use the same configuration when auditing a run that selected different
families or datasets. The extraction command itself never combines tables.

## Python use and compatibility

The runner also accepts a configuration dictionary, making later experiments
possible without editing the original YAML:

```python
from training.config import load_config
from training.extraction import extract_configured_features

config = load_config()
frames = extract_configured_features(config, "data/features_configured/baseline")
```

Direct dataset extractor calls opt in with the keyword-only `preprocessing`
argument. Pass a resolved family mapping, or a partial mapping filled with
shared grayscale-only defaults:

```python
from microalgae.imaging.features import lbp_features
from microalgae.imaging.preprocessing.configured import resolve_preprocessing

frame = lbp_features.extract_features(
    "data/raw", config["dataset"]["classes"], "data/features_configured/raw_lbp",
    preprocessing=resolve_preprocessing(config, "lbp"), standardize=False,
)
```

Direct wrappers preserve their old output-directory behavior; the new-directory
safety rule belongs to the runner. Omitting `preprocessing`, or passing None,
preserves legacy preprocessing, numerical defaults, and preview paths. With
configured PC extraction, preprocessing settings take precedence over the old
`resize` argument; `ddof` and Phasepack parameter overrides remain independent.
The existing generic `preprocess_image` function is unchanged. Training now has
a separate [nested experiment entry point](nested_experiments.md).

To compare preprocessing combinations later, copy the configuration dictionary
or YAML, change explicit family overrides, and use a separate output directory
for each run. Keep dataset identity, feature choices, splits, and scaling
procedures controlled. Optuna, parent grouping, and model comparison are not
part of this command.
