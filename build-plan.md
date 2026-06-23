# cosmock Object-Oriented Reorganization Plan

## Summary

`cosmock` is a kappa-first mock-field generator. The public workflow is object
oriented where state matters:

```python
import cosmock as cm

cal = cm.KappaCalibration.from_maps(kappa_maps, nside=512, cl_ng=cl_ng, pixwin=pixwin)
model = cm.MockModel.fit(cal, transform="gptg", order=3)
mocks = model.sample(n_mocks=100, seed=1234, apply_pixwin=True)
report = model.validate(mocks)
```

Stateless numerical kernels remain lower-level functions so formulas are easy to
test against the paper and the legacy `/home/denniswu28/Extended_Lognormal`
implementation.

## Public API

Keep the root namespace narrow:

- `KappaCalibration`
- `MockModel`
- `MockValidationReport`
- `SpectrumDiagnostics`

Do not expose root-level helpers such as `fit_gptg`, `generate_mocks`,
`create_mock`, `Gn`, or `C_NG_to_C_G`.

## Package Structure

```text
src/cosmock/
  __init__.py
  calibration.py
  transforms.py
  fitting.py
  spectra.py
  generation.py
  healpix.py
  plotting.py
  datasets/
    gower_st.py
  util/
    optional.py
    quadrature.py
    statistics.py
    validation.py
```

`debiasing.py` is out of scope for this pass.

## Object Model

- `KappaCalibration.from_maps(...)` validates maps, estimates or stores spectra,
  and computes Gaussianized calibration data.
- `KappaCalibration.fit_transform(...)` returns a `GPTGTransformSet`.
- `GPTGTransform` and `GPTGTransformSet` hold fitted point-transform parameters
  and expose `.evaluate(...)`.
- `GPTGTransformSet.to_latent_spectra(...)` converts target kappa spectra to
  latent Gaussian spectra.
- `MockModel.fit(...)` uses the calibration and transform set, then stores
  fitted parameters, target spectra, latent spectra, generation metadata, and
  diagnostics state.
- `MockModel.sample(...)`, `.validate(...)`, and `.diagnostics(...)` remain the
  supported user workflow.

## Dependencies

Base dependencies stay limited to:

- `numpy`
- `scipy`
- `joblib`

Optional extras stay lazy:

- `healpy`: `cosmock[healpix]`
- `matplotlib`: `cosmock[plot]`
- `pyccl`: `cosmock[ccl]`
- `torch` plus `healpy`: `cosmock[torch]`

## Acceptance

```bash
python -m compileall src/cosmock tests
PYTHONPATH=src python -m pytest -q
```

After editable install:

```bash
python -m pytest
```

Optional HEALPix check:

```bash
python -m pip install -e ".[dev,healpix,plot]"
python -m pytest
```
