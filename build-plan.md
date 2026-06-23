# Comprehensive cosmock Migration Plan

## Summary
Build `cosmock` as a kappa-first mock-field generator: calibrate from input convergence/kappa maps and generate statistically matched mock kappa maps. Public docs may mention projected cosmological density fields, but v1 does not generate 3D density volumes.

Use the teammate’s three-function spine as the implementation story:

1. fit transform parameters from maps.
2. convert target non-Gaussian spectra `C^kappa` / `C^delta` to latent Gaussian spectra `C^x`.
3. create mocks from `C^x` and transform parameters.

Copy code from `/home/denniswu28/Extended_Lognormal`; do not edit or move that legacy repo.

## Public API
Expose both high-level and lower-level APIs:

```python
import cosmock as cm

cal = cm.KappaCalibration.from_maps(kappa_maps, nside=512, cl_ng=cl_ng, pixwin=pixwin)
model = cm.fit_gptg(cal, order=3)
mocks = cm.generate_mocks(model, n_mocks=100, seed=1234, apply_pixwin=True)
```

```python
model = cm.MockModel.fit(cal, transform="gptg", order=3)
mocks = model.sample(n_mocks=10, seed=42)
```

```python
transform_params = cm.fit_transform(cal, order=3)
cl_x = cm.target_cls_to_latent_cls(cal.cl_ng, transform_params, order=3)
mock = cm.create_mock(cl_x, transform_params, nside=512, seed=42)
```

Use `transform_params`, not `lambda`, because `lambda` is reserved in Python. `MockModel.sample()` must return `(n_mocks, n_bins, n_pix)`.

## Package Structure
Create:

```text
cosmock/
  pyproject.toml
  README.md
  LICENSE
  CITATION.cff
  AGENTS.md
  src/cosmock/
    __init__.py
    calibration.py
    transforms.py
    fitting.py
    spectra.py
    generation.py
    debiasing.py
    validation.py
    plotting.py
    healpix.py
    quadrature.py
    datasets/
      __init__.py
      gower_st.py
  tests/
  examples/basic_kappa_to_mocks.ipynb
  docs/agent_harness/
```

Map legacy code with minimal behavior changes:

- `fitter/Gn.py` + `gauss_hermite.py` -> `transforms.py`, `quadrature.py`.
- `fitter/fitter.py` -> `fitting.py`.
- `fitter/auxiliary_functions.py` + `utils/histograms.py` -> `calibration.py`.
- `fitter/transform_cls.py` + `variance_calc.py` -> `spectra.py`.
- `fitter/mocker.py` -> `generation.py`.
- `fitter/debiaser.py` -> `debiasing.py`.
- `fitter/plots.py` -> `plotting.py`.
- `utils/transforms.py` -> `healpix.py`.
- `utils/GowerSt_map_maker.py` -> `datasets/gower_st.py`.

## Implementation Details
`KappaCalibration` should store maps, `nside`, `n_bins`, `n_pix`, Gaussianized samples, binned transform data, target `cl_ng`, and optional `pixwin`.

`MockModel` should store `transform`, `order`, `transform_params`, `cl_ng`, `cl_x`, `nside`, optional `pixwin`, and generation defaults.

Validate shapes early:

- `kappa_maps`: `(n_bins, n_pix)` or one map coercible to `(1, n_pix)`.
- `cl_ng`: `(n_bins, n_bins, lmax + 1)`.
- `pixwin`: `(lmax + 1,)`.

If `cl_ng` is omitted, estimate spectra only when `healpy` is installed; otherwise raise a clear install error.

Core dependencies: `numpy`, `scipy`, `joblib`.

Extras:

```text
healpix = ["healpy"]
plot = ["matplotlib"]
ccl = ["pyccl"]
torch = ["torch", "healpy"]
dev = ["pytest", "pytest-cov", "ruff", "jupyter", "nbmake", "build"]
```

`import cosmock` must work without optional dependencies.

## Docs And Agent Harness
Write `README.md` around the kappa-to-mocks workflow, not GPTG fitting as the primary identity.

Use BSD-3-Clause for `LICENSE`.

Create a minimal `CITATION.cff` referencing arXiv:2411.04759 without inventing missing author metadata.

Add root `AGENTS.md` with migration rules, optional dependency policy, validation commands, and “do not rewrite scientific formulas before tests” guidance.

Because `.agents/` is currently read-only, put harness files in `docs/agent_harness/`:

- `migration_checklist.md`
- `feature_tracks.md`
- `review_checklist.md`

## Example And Tests
Create `examples/basic_kappa_to_mocks.ipynb` as a data-path template, not a private-data notebook. It should show expected input shapes and use only the new `cosmock` API.

Tests:

- core import with only required dependencies.
- calibration from maps plus supplied spectra.
- finite GPTG transform fitting on deterministic toy data.
- target-to-latent spectra conversion returns finite arrays with expected shape.
- clear optional dependency errors for HEALPix-only paths.
- seeded mock generation is deterministic and finite when `healpy` is installed.

Acceptance commands:

```bash
python -m pip install -e ".[dev]"
python -m compileall src/cosmock tests
python -m pytest
```

Optional HEALPix check:

```bash
python -m pip install -e ".[dev,healpix,plot]"
python -m pytest
```
