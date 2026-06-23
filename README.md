# cosmock

`cosmock` generates statistically matched mock cosmological kappa fields from
input convergence maps. The input kappa map is the calibration object; the
outputs are mock kappa maps that are designed to reproduce the calibrated
one-point distribution and angular spectra.

The package is kappa-first. It can be useful for projected cosmological density
field workflows, but v1 does not generate 3D density volumes.

## Core Workflow

```python
import numpy as np
import cosmock as cm

kappa_maps = np.load("path/to/kappa_maps.npy")  # (n_bins, n_pix)
cl_ng = np.load("path/to/target_cls.npy")       # (n_bins, n_bins, lmax + 1)
pixwin = np.load("path/to/pixwin.npy")          # optional, (lmax + 1,)

cal = cm.KappaCalibration.from_maps(
    kappa_maps,
    nside=512,
    cl_ng=cl_ng,
    pixwin=pixwin,
)

model = cm.MockModel.fit(cal, transform="gptg", order=3)
mocks = model.sample(n_mocks=100, seed=1234, apply_pixwin=True)
report = model.validate(mocks)
```

This object workflow is the supported public API. User-facing operations that
carry state live on domain objects such as `KappaCalibration` and `MockModel`.
Stateless numerical helpers remain lower-level in implementation modules such
as `cosmock.fitting`, `cosmock.spectra`, and `cosmock.util` for advanced
development and tests.

`MockModel.fit()` supports GPTG orders 2 and 3 as the tested v1 science path.
`model.sample()` always returns mock maps with shape `(n_mocks, n_bins, n_pix)`.

## Installation

The core package keeps optional scientific dependencies out of the base install:

```bash
pip install -e .
```

Core dependencies:

- `numpy`
- `scipy`
- `joblib`

Optional extras:

```bash
pip install -e ".[healpix]"
pip install -e ".[plot]"
pip install -e ".[ccl]"
pip install -e ".[torch]"
pip install -e ".[dev]"
```

`import cosmock` works without `healpy`, `matplotlib`, `pyccl`, or `torch`.
Features that need those packages raise an error with the matching extra.

## Input Shapes

- `kappa_maps`: `(n_bins, n_pix)` or one map as `(n_pix,)`.
- `cl_ng`: `(n_bins, n_bins, lmax + 1)`.
- `pixwin`: `(lmax + 1,)`.

If `cl_ng` is not supplied, `KappaCalibration.from_maps` estimates spectra with
`healpy`, so install `cosmock[healpix]`.

## Examples

- `examples/cosmock_tutorial.ipynb`: runnable tutorial with synthetic kappa maps and plots.
- `examples/basic_kappa_to_mocks.ipynb`: template for local data paths.

## Development

```bash
python -m pip install -e ".[dev]"
python -m compileall src/cosmock tests
python -m pytest
```

Optional HEALPix workflow checks:

```bash
python -m pip install -e ".[dev,healpix,plot]"
python -m pytest
```
