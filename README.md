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

The repository includes a small example data set under `data/`. The code
below fits a `G3` transformation to the included kappa map and generates
one mock map cube.

```python
import numpy as np

from cosmock import fit_parameters, generate_mocks

path_to_map = "data/Kappa_Gower_St_ID_44_DESy3_tomography_Nside_256.npy"
path_to_cl = "data/UNBIASED_3point75nsideminus1_Cls_NG_Gower_St_ID_44.npy"
path_to_pixwin = "data/pixwin_256.npy"

kappa_map = np.load(path_to_map)
cl_ng = np.load(path_to_cl)
pixwin = np.load(path_to_pixwin)

order = 3
n_mocks = 1

params = fit_parameters(kappa_map, cl_ng, order)
mocks = generate_mocks(params, n_mocks, pixwin=pixwin)

print(mocks.shape)
```

Expected progress messages:

```text
Done fitting the G3 parameters
Done finding the power spectrum of the underlying gaussian random field
```

Expected output shape:

```text
(1, N_bins, N_pix)
```

where `N_bins` is the number of tomographic bins in `kappa_map` and
`N_pix` is the number of HEALPix pixels per map. For the included data,
`N_pix` corresponds to `nside=256`.

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

The formal software citation for `cosmock` will be provided by the
forthcoming software paper. Until then, cite the repository version or
commit hash used in your analysis so that results can be reproduced.

This package uses ideas from the paper
[Fast Generation of Weak Lensing Maps with Analytical Point Transformation
Functions](https://arxiv.org/abs/2411.04759), but that paper is background
for the method rather than the software citation.
