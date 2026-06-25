# cosmock

`cosmock` generates mock HEALPix maps for scalar random fields when you can
specify:

1. the field's one-point distribution from example maps, and
2. the field's two-point statistics as angular power spectra.

The bundled data are weak-lensing convergence (`kappa`) maps, but the package
boundary is intentionally generic: density, convergence, or another scalar
field can use the same workflow if the input maps and spectra follow the
documented shapes.

The implementation uses generalized point-transformed Gaussian fields,
following the method in
[Fast Generation of Weak Lensing Maps with Analytical Point Transformation
Functions](https://arxiv.org/abs/2411.04759).

## Installation

From the repository root:

```bash
python -m pip install -e .
```

For development:

```bash
python -m pip install -r requirements-dev.txt
```

## Quickstart

The standard workflow has two public functions:

```python
import numpy as np

import cosmock

field_maps = np.load("data/Kappa_Gower_St_ID_44_DESy3_tomography_Nside_256.npy")
cl_field = np.load("data/UNBIASED_3point75nsideminus1_Cls_NG_Gower_St_ID_44.npy")
pixwin = np.load("data/pixwin_256.npy")

fit = cosmock.fit_field_model(field_maps, cl_field, order=3)
mocks = cosmock.generate_field_mocks(fit, n_mocks=1, seed=123, pixwin=pixwin)

print(mocks.shape)
```

Expected output shape:

```text
(1, n_bins, n_pix)
```

where `n_bins` is the number of input fields or tomographic bins and `n_pix`
is the number of HEALPix pixels per map.

## Public API

### `fit_field_model(field_maps, cl_field, order, *, ...)`

Fits the point-transformation parameters that match the one-point statistics
of the input scalar-field maps, then converts the target non-Gaussian field
spectra into spectra for the latent Gaussian fields.

- `field_maps`: array with shape `(n_bins, n_pix)` containing finite HEALPix
  scalar-field maps.
- `cl_field`: array with shape `(n_bins, n_bins, n_ell)` containing target
  angular power spectra for the same fields.
- `order`: transformation order, currently `2` or `3`.
- returns: `FieldMockFit`, a frozen dataclass containing transform parameters,
  latent spectra, HEALPix resolution, and generation limits.

### `generate_field_mocks(fit, n_mocks, *, seed=None, pixwin=None)`

Samples latent Gaussian HEALPix maps, applies the fitted point transformation,
and applies a HEALPix pixel window.

- `fit`: the `FieldMockFit` returned by `fit_field_model`.
- `n_mocks`: number of mock map cubes to generate.
- `seed`: optional NumPy random seed, `SeedSequence`, or `Generator`.
- `pixwin`: optional HEALPix pixel window indexed by multipole. If omitted,
  `healpy.pixwin` is computed from the fitted resolution.
- returns: array with shape `(n_mocks, n_bins, n_pix)`.

## Development Checks

```bash
python -m compileall cosmock tests
python -m pytest -q
python -m ruff check cosmock tests
python -m build --no-isolation
```

## Citation

The formal software citation for `cosmock` will be provided by the forthcoming
software paper.

This package uses the method from the paper
[Fast Generation of Weak Lensing Maps with Analytical Point Transformation
Functions](https://arxiv.org/abs/2411.04759).
