# Log-Gabor feature extraction

The default extractor uses a fixed-profile Python port of the [LogGabor MATLAB
toolbox cited by Sanchez et al.](https://doi.org/10.6084/m9.figshare.2067504.v1),
not the similarly named Python LogGabor package. The port uses NumPy and makes
no network requests. The old `default_param.py` remains a legacy reference and
does not control this bank.

## Filter bank

Every response stays on the project's `256 x 256` pixel grid. The bank has four
scales and six orientations, following the counts in [Sanchez et al. (2019)](https://doi.org/10.7717/peerj.6770),
pp. 9-10. Its nominal frequencies and frequency-domain directions are:

| Scale | Nominal frequency (cycles/pixel) | Directions (degrees) |
| --- | --- | --- |
| 1 | 0.5 | 0, 30, 60, 90, 120, 150 |
| 2 | 0.25 | 15, 45, 75, 105, 135, 165 |
| 3 | 0.125 | 0, 30, 60, 90, 120, 150 |
| 4 | 0.0625 | 15, 45, 75, 105, 135, 165 |

These are the generating centers, not a guarantee about the maximum sampled
coefficient after boundary folding and normalization. Directions use the
toolbox's image-coordinate convention: zero points along positive columns.

The selected MATLAB settings are `Nf=4`, `No=6`, `csf=-1`, and:

```text
param = [0,0.5,0,2,0,0,0,0,0,1,1,0,3,5,6,0,0,2,0,3,2,0,3]
```

This retains the toolbox's octave spacing, even-scale rotation, complex phases,
first-scale half-pixel phase construction, periodic boundary folding, global
filter normalization, and unit-coefficient amplitude calibration. Downsampling
and filter-window truncation are disabled. The residual low-pass participates
in normalization but is excluded from the feature vector. Bandpass DC values
are set to exactly zero rather than retaining the source's tiny approximation.

The private bank module records the archive URL, SHA256, original authors, and
upstream GNU GPL notice. The port includes the source's amplitude-calibration
calculation, but does not expose reconstruction or denoising functionality.

## Calculation and preprocessing

```python
from microalgae.imaging.features.gabor_features import (
    extract_features,
    extract_gabor_statistics,
)

# gray_256 is a finite, real 256 x 256 grayscale array; roi selects pixels.
values = extract_gabor_statistics(gray_256, mask=roi)

# Existing four positional arguments are unchanged. Returns the saved frame.
frame = extract_features(dataset_dir, class_subfolders, features_dir, False)
```

`extract_gabor_statistics` performs no loading, resizing, segmentation, saving,
or input-intensity normalization. Its optional mask selects pixels **after**
filtering. With no mask, statistics use every pixel. A supplied `lg` object is
still accepted through `lg=...`; it must provide `linear_pyramid(image)` with
finite responses shaped `(H, W, 6, 4)`. The compatibility wrapper
`extract_gabor_features` retains its six positional arguments.

The dataset wrapper preserves the existing preprocessing: grayscale loading,
aspect-preserving padded resize to `256 x 256`, `segment_image` with
`dilation_size=4`, and input masking before filtering. The same foreground mask
also selects the subsequent statistics. Empty masks are rejected. Actual zero
responses inside the foreground remain valid data.

For each scale, the six complex response magnitudes are summed at each pixel.
The pooled image supplies 13 first-order statistics and 19 GLCM statistics at
each of three distances and four directions. The result is
`4 * (13 + 19 * 3 * 4) = 964` features, in the existing per-scale column order.

## Statistical conventions

The feature families and masked measurements follow [Bueno et al. (2017)](https://doi.org/10.3390/app7080753),
Section 4 and Tables 2-4, cited by Sanchez. Some printed statistical formulas
are ambiguous or inconsistent, so the implementation uses these explicit
conventions rather than claiming literal numerical reproduction:

- First-order mean, minimum, maximum, population variance, percentiles and
  interquartile range use the selected response values directly.
- The 255-bin histogram spans the selected response range. Mode is the center
  of its most populated bin; ties choose the lowest bin. Constant regions
  return their actual value. Histogram entropy uses normalized counts and
  base-2 logarithms.
- Skewness and Pearson kurtosis use SciPy's default biased estimators. Constant
  and single-pixel regions use zero for both undefined standardized moments.
- GLCMs quantize the foreground range to 16 levels. Constant foreground maps
  to level zero. Distances are 1, 3 and 5 pixels; directions are 0, 45, 90 and
  135 degrees. Each matrix is symmetric and normalized over pairs whose
  **two endpoints** belong to the mask. Borders do not wrap.
- An offset without valid pairs contributes 19 zeros. Valid zero-valued
  foreground pairs are not removed.
- Existing second-order formulas retain their `1e-10` numerical epsilon and
  base-2 entropy convention. Energy is `sum(c**2)`. Sum variance is centered on
  sum entropy; difference variance is the raw second moment of `abs(i-j)`.
  The two homogeneity denominators are `1+(i-j)**2` and `1+abs(i-j)`.

The 255-bin default, histogram-based mode, and Pearson kurtosis deliberately
replace the previous 256-bin entropy, raw-value mode, and excess kurtosis.
Ambiguous second-order formulas have not been silently redefined. Standalone
GLCM extraction is unaffected.

## Outputs and verification limits

`Gabor_features.csv` retains its 964 feature columns followed by `Class`.
Values are unstandardized; no feature selection or model fitting occurs here.
Rows retain the supplied class order and valid images' directory order.
Unsupported files are ignored, and extraction failures identify the source
image instead of silently dropping its row.

Optional uint8 input previews are saved under
`data/success/gabor/<class>/<image-stem>.jpg`. Failed writes raise errors.
The old, incorrectly targeted `data/success/efd` files are not deleted.

The 964-feature count matches Bueno's Log-Gabor family. The correlation-based
reduction to 177 features reported by Sanchez is a separate, dataset-dependent
selection step and is not implemented by this extractor.

Analytical tests cover Gaussian samples, boundary folding, spectral symmetry,
normalization, amplitude calibration, impulse and sinusoidal responses, masks,
and dataset exports. No MATLAB/Octave runtime was available for direct upstream
comparison. Sanchez does not disclose every toolbox setting or statistical
convention, so this configuration is not confirmed reproduction of the authors'
experiment.

Adopting this bank and the corrected statistics changes feature values. Existing
feature CSVs and models are not modified automatically: regenerate Log-Gabor
features and retrain models before comparing results. Align image identities
before combining independently generated feature tables.
