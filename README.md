# cosmock

`cosmock` generates mock weak-lensing convergence (`kappa`) maps from an
input kappa map and its angular power spectra. It implements a
generalized point-transformed Gaussian mock-generation workflow using ideas
from the paper
[Fast Generation of Weak Lensing Maps with Analytical Point Transformation
Functions](https://arxiv.org/abs/2411.04759).

The standard user path is:

1. load a tomographic kappa map and its non-Gaussian `C_ell` spectra,
2. fit the point-transformation parameters with `fit_parameters`,
3. sample one or more mock kappa-map cubes with `generate_mocks`.

Kappa is dimensionless lensing convergence. The spectra passed to and
returned by this package are angular power spectra of those fields.

## Installation

From the repository root, install the package in editable mode:

```bash
python -m pip install -e .
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

Generates mock kappa-map cubes from fitted parameters.

- `params`: the object returned by `fit_parameters`.
- `Nmocks`: number of mock cubes to generate.
- `pixwin`: optional HEALPix pixel window array. If omitted, `healpy`
  computes it from the fitted map resolution.
- returns: array with shape `(Nmocks, N_bins, N_pix)`.

## Citation

This package uses ideas from the paper
[Fast Generation of Weak Lensing Maps with Analytical Point Transformation
Functions](https://arxiv.org/abs/2411.04759) as background.
