# Feature Tracks

## Calibration

Map validation, empirical CDFs, Gaussianized samples, binned transform data, and
optional spectrum estimation.

## Fitting

GPTG parameter fitting from calibration data. Preserve constrained G2/G3
behavior until validated against legacy notebooks.

## Spectra

Target non-Gaussian spectra to latent Gaussian spectra. Focus tests on shape,
finite output, and positive-definite diagnostics before changing algorithms.

## Generation

Gaussian HEALPix draws, pixel window application, mock kappa map output, and
seeded deterministic sampling.

## Validation And Plotting

Histogram and spectra comparisons. Plotting must remain optional.

## Datasets

Project-specific loaders such as Gower Street. These should never be required
for the base package import.

