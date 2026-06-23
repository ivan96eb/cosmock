**Review Plan**

Start with behavior and public API risks before style.

1. **Public API Contract**
   - Review [src/cosmock/__init__.py](/home/denniswu28/cosmock/src/cosmock/__init__.py:1), [README.md](/home/denniswu28/cosmock/README.md:1), and [examples/basic_kappa_to_mocks.ipynb](/home/denniswu28/cosmock/examples/basic_kappa_to_mocks.ipynb:1).
   - Confirm the root namespace only exposes:
     `KappaCalibration`, `MockModel`, `MockValidationReport`, and `SpectrumDiagnostics`.
   - Confirm the supported user path is:
     `KappaCalibration.from_maps()` -> `MockModel.fit()` -> `model.sample()` -> `model.validate()`.

2. **Object-Oriented Workflow Review**
   - Check that user-facing operations carrying state are methods:
     `KappaCalibration.fit_transform()`, `GPTGTransform.evaluate()`,
     `GPTGTransformSet.to_latent_spectra()`, `MockModel.sample()`,
     `MockModel.validate()`, and `MockModel.diagnostics()`.
   - Check that stateless numerical formulas remain lower-level functions in
     implementation modules or `cosmock.util`.

3. **Legacy Parity Review**
   Compare migrated numerical behavior against the legacy source:

   - `transforms.py` vs `fitter/Gn.py`
   - `util/quadrature.py` vs `fitter/gauss_hermite.py`
   - `fitting.py` vs `fitter/fitter.py`
   - `util/statistics.py` and `calibration.py` vs `fitter/auxiliary_functions.py` and `utils/histograms.py`
   - `spectra.py` vs `fitter/transform_cls.py` and `fitter/variance_calc.py`
   - `generation.py` vs `fitter/mocker.py`
   - `plotting.py` vs `fitter/plots.py`
   - `healpix.py` vs `utils/transforms.py`
   - `datasets/gower_st.py` vs `utils/GowerSt_map_maker.py`

4. **Science-Critical Review Order**
   1. [src/cosmock/transforms.py](/home/denniswu28/cosmock/src/cosmock/transforms.py:1): GPTG formulas and transform objects.
   2. [src/cosmock/fitting.py](/home/denniswu28/cosmock/src/cosmock/fitting.py:1): constrained/unconstrained parameter fitting.
   3. [src/cosmock/spectra.py](/home/denniswu28/cosmock/src/cosmock/spectra.py:1): target-to-latent spectra conversion.
   4. [src/cosmock/generation.py](/home/denniswu28/cosmock/src/cosmock/generation.py:1): sampling, seeding, and pixwin behavior.
   5. [src/cosmock/calibration.py](/home/denniswu28/cosmock/src/cosmock/calibration.py:1): calibration state and method boundaries.
   6. [src/cosmock/util/validation.py](/home/denniswu28/cosmock/src/cosmock/util/validation.py:1): stricter checks and report objects.

5. **Acceptance Gate**
   ```bash
   python -m compileall src/cosmock tests
   PYTHONPATH=src python -m pytest -q
   ```

   After editable install:

   ```bash
   python -m pytest
   ```

   Optional HEALPix gate when available:

   ```bash
   python -m pip install -e ".[dev,healpix,plot]"
   python -m pytest
   ```
