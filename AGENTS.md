# Repository Guidelines

## Project Identity

`cosmock` is a kappa-first mock-field generator. The public story is: calibrate
from input convergence/kappa maps, fit a transformation, convert target spectra
to latent Gaussian spectra, and generate statistically matched mock kappa maps.

Do not frame the package primarily as a GPTG fitting library. GPTG is an
implementation detail exposed through a small set of lower-level functions.

## Migration Rules

- Copy or adapt code from `/home/denniswu28/Extended_Lognormal`; do not edit or
  move that legacy repository.
- Preserve numerical formulas until focused tests exist for the behavior being
  changed.
- Avoid root-level public names such as `fitter`, `mocker`, `Gn`, or lower-level
  numerical helpers.
- Prefer the standardized object API: `KappaCalibration.from_maps(...)`,
  `MockModel.fit(...)`, `model.sample(...)`, and `model.validate(...)`.
- Keep lower-level helpers in implementation modules such as `cosmock.fitting`,
  `cosmock.spectra`, and `cosmock.generation` for tests and advanced users.

## Optional Dependencies

Base imports must require only `numpy`, `scipy`, and `joblib`.

Keep optional dependencies behind extras and lazy import guards:

- `healpy`: `cosmock[healpix]`
- `matplotlib`: `cosmock[plot]`
- `pyccl`: `cosmock[ccl]`
- `torch` plus `healpy`: `cosmock[torch]`

If a feature needs an optional dependency, raise an actionable error naming the
extra to install.

## Validation Commands

Run these before handing off broad scaffold changes:

```bash
python -m compileall src/cosmock tests
python -m pytest
```

For HEALPix-specific work:

```bash
python -m pip install -e ".[dev,healpix,plot]"
python -m pytest
```
