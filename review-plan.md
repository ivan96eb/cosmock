**Review Plan**

Start with a line-by-line review in risk order, not file order.

1. **Public API Contract**
   - Review [src/cosmock/__init__.py](/home/denniswu28/cosmock/src/cosmock/__init__.py:1), [README.md](/home/denniswu28/cosmock/README.md:1), and [examples/basic_kappa_to_mocks.ipynb](/home/denniswu28/cosmock/examples/basic_kappa_to_mocks.ipynb:1).
   - Confirm the only top-level user path is:
     `KappaCalibration.from_maps()` -> `MockModel.fit()` -> `model.sample()` -> `model.validate()`.
   - Known issue to check: [AGENTS.md](/home/denniswu28/cosmock/AGENTS.md:19) still mentions old top-level API names and `cosmock.Gn`, which now conflicts with the standardized API.

2. **Legacy Parity Review**
   Compare each migrated file against the legacy source:

   - `transforms.py` vs `fitter/Gn.py`
   - `quadrature.py` vs `fitter/gauss_hermite.py`
   - `fitting.py` vs `fitter/fitter.py`
   - `calibration.py` vs `fitter/auxiliary_functions.py` and `utils/histograms.py`
   - `spectra.py` vs `fitter/transform_cls.py` and `fitter/variance_calc.py`
   - `generation.py` vs `fitter/mocker.py`
   - `debiasing.py` vs `fitter/debiaser.py`
   - `plotting.py` vs `fitter/plots.py`
   - `healpix.py` vs `utils/transforms.py`
   - `datasets/gower_st.py` vs `utils/GowerSt_map_maker.py`

   For each function, classify the change as:
   - `unchanged port`
   - `renamed only`
   - `intentional API wrapper`
   - `behavior changed`
   - `new code`

3. **Science-Critical Review Order**
   Review these first, line by line:

   1. [src/cosmock/transforms.py](/home/denniswu28/cosmock/src/cosmock/transforms.py:1): GPTG formulas.
   2. [src/cosmock/fitting.py](/home/denniswu28/cosmock/src/cosmock/fitting.py:1): constrained/unconstrained parameter fitting.
   3. [src/cosmock/spectra.py](/home/denniswu28/cosmock/src/cosmock/spectra.py:1): `C_NG -> C_G` conversion.
   4. [src/cosmock/generation.py](/home/denniswu28/cosmock/src/cosmock/generation.py:1): sampling, seeding, pixwin behavior.
   5. [src/cosmock/calibration.py](/home/denniswu28/cosmock/src/cosmock/calibration.py:1): empirical CDF, Gaussianization, shape handling.
   6. [src/cosmock/validation.py](/home/denniswu28/cosmock/src/cosmock/validation.py:1): new stricter checks and report objects.

4. **Review Commands**
   Use these while reviewing:

   ```bash
   nl -ba src/cosmock/generation.py
   nl -ba /home/denniswu28/Extended_Lognormal/fitter/mocker.py

   diff -u \
     /home/denniswu28/Extended_Lognormal/fitter/mocker.py \
     src/cosmock/generation.py
   ```

   For searched API drift:

   ```bash
   rg -n "fit_gptg|generate_mocks|fit_transform|target_cls_to_latent_cls|create_mock|Gn" .
   ```

5. **Output Format**
   For each reviewed file, produce a short ledger:

   ```text
   File: src/cosmock/spectra.py
   Legacy source: fitter/transform_cls.py, fitter/variance_calc.py
   Status: behavior changed / API wrapped
   Findings:
   - line X: ...
   Intentional changes:
   - ...
   Test coverage:
   - covered by ...
   Missing tests:
   - ...
   Decision:
   - accept / fix before release / needs science review
   ```

6. **Acceptance Gate**
   After review fixes, require:

   ```bash
   PYTHONPATH=src python -m compileall src/cosmock tests
   PYTHONPATH=src python -m pytest -q
   ```

   Optional HEALPix gate when available:

   ```bash
   python -m pip install -e ".[dev,healpix,plot]"
   python -m pytest -q
   ```