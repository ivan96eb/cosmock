# cosmock

`cosmock` generates mock weak-lensing convergence (`kappa`) maps from an
input kappa map and its angular power spectra. It implements a
generalized point-transformed Gaussian mock-generation workflow using ideas
from the paper
[Fast Generation of Weak Lensing Maps with Analytical Point Transformation
Functions](https://arxiv.org/abs/2411.04759).

The standard user path is:

1. Load a tomographic kappa map and its non-Gaussian `C_ell` spectra,
2. Fit the point-transformation parameters with `fit_parameters`,
3. Sample one or more mock kappa maps with `generate_mocks`.

Kappa is dimensionless lensing convergence. The spectra passed to and
returned by this package are angular power spectra of those fields.

## Installation

```bash
pip install cosmock
```

## Quickstart

The repository includes a small example data set under `data/`. The notebook `Quickstart.ipynb` provides a minimal working example showing how to use this example dataset to run the full pipeline.

## Public API

### `fit_parameters(maps, Cl_delta, N)`

Fits the nonlinear transformation parameters for each tomographic bin and
converts the input non-Gaussian spectra into spectra for the latent
Gaussian field.

- `maps`: array with shape `(N_bins, N_pix)` containing dimensionless
  kappa maps.
- `Cl_delta`: array with shape `(N_bins, N_bins, N_ell)` containing the
  target kappa angular power spectra.
- `N`: transformation order. The current implementation supports `2` and
  `3`.
- returns: a `NonlinParameters` object containing fitted parameters,
  Gaussian spectra, transformation order, and the original map shape.

### `generate_mocks(params, Nmocks, pixwin=None)`

Generates mock kappa maps from fitted parameters.

- `params`: the object returned by `fit_parameters`.
- `Nmocks`: number of mocks to generate.
- `pixwin`: optional HEALPix pixel window array. If omitted, `healpy`
  computes it from the fitted map resolution.
- returns: array with shape `(Nmocks, N_bins, N_pix)`.

## Citation

This package uses the method from the paper
[Fast Generation of Weak Lensing Maps with Analytical Point Transformation
Functions](https://arxiv.org/abs/2411.04759) as background.
