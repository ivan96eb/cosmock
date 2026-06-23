from __future__ import annotations

import numpy as np
import pytest

import cosmock as cm
from cosmock.transforms import GPTGTransformSet
from cosmock.util.optional import OptionalDependencyError
from cosmock.util.validation import MockValidationReport


def _toy_maps(n_bins=2, nside=2):
    n_pix = 12 * nside * nside
    x = np.linspace(-1.0, 1.0, n_pix)
    maps = []
    for i in range(n_bins):
        maps.append(0.01 * (i + 1) * x + 0.002 * np.sin((i + 1) * np.pi * x))
    return np.asarray(maps)


def _toy_cls(n_bins=2, lmax=6):
    ell = np.arange(lmax + 1)
    cl = np.zeros((n_bins, n_bins, lmax + 1), dtype=float)
    base = np.zeros_like(ell, dtype=float)
    base[2:] = 1e-5 / (ell[2:] + 1) ** 2
    for i in range(n_bins):
        cl[i, i] = (i + 1) * base
    if n_bins > 1:
        for i in range(n_bins):
            for j in range(i):
                cl[i, j] = 0.25 * np.sqrt(cl[i, i] * cl[j, j])
                cl[j, i] = cl[i, j]
    cl[:, :, 0] = 1e-20 * np.eye(n_bins)
    cl[:, :, 1] = 1e-20 * np.eye(n_bins)
    return cl


def test_core_import_exposes_public_api():
    assert cm.KappaCalibration is not None
    assert cm.MockModel is not None
    assert cm.MockValidationReport is MockValidationReport
    assert set(cm.__all__) == {
        "KappaCalibration",
        "MockModel",
        "MockValidationReport",
        "SpectrumDiagnostics",
    }
    assert "fit_transform" not in cm.__all__
    assert "target_cls_to_latent_cls" not in cm.__all__
    assert "create_mock" not in cm.__all__
    assert not hasattr(cm, "fit_gptg")
    assert not hasattr(cm, "generate_mocks")
    assert not hasattr(cm, "create_mock")
    assert not hasattr(cm, "Gn")
    assert not hasattr(cm, "C_NG_to_C_G")


def test_calibration_from_maps_with_supplied_spectra():
    maps = _toy_maps()
    cl_ng = _toy_cls()

    cal = cm.KappaCalibration.from_maps(
        maps,
        nside=2,
        cl_ng=cl_ng,
        n_fit_bins=12,
    )

    assert cal.maps.shape == maps.shape
    assert cal.n_bins == 2
    assert cal.n_pix == maps.shape[1]
    assert cal.cl_ng.shape == cl_ng.shape
    assert len(cal.gaussianized_samples) == 2
    assert len(cal.binned_x) == 2
    assert len(cal.binned_y) == 2


def test_mock_model_fit_returns_finite_gptg_parameters():
    cal = cm.KappaCalibration.from_maps(
        _toy_maps(),
        nside=2,
        cl_ng=_toy_cls(),
        n_fit_bins=12,
    )

    model = cm.MockModel.fit(cal, order=3, constrained=False, n_jobs=1, quad_order=2)
    transform_params = model.transform_params

    assert transform_params.shape == (2, 3)
    assert np.all(np.isfinite(transform_params))


def test_mock_model_fit_stores_expected_fields():
    cal = cm.KappaCalibration.from_maps(
        _toy_maps(),
        nside=2,
        cl_ng=_toy_cls(),
        n_fit_bins=12,
    )

    model = cm.MockModel.fit(
        cal,
        order=2,
        constrained=False,
        n_jobs=1,
        quad_order=2,
    )

    assert model.transform == "gptg"
    assert model.order == "2"
    assert model.n_bins == 2
    assert model.n_pix == cal.n_pix
    assert model.lmax == cal.cl_ng.shape[-1] - 1
    assert model.transform_params.shape == (2, 2)
    assert model.transform_set is not None
    assert model.transform_set.order == "2"
    np.testing.assert_allclose(model.transform_set.transform_params, model.transform_params)
    assert model.cl_ng.shape == cal.cl_ng.shape
    assert model.cl_x.shape == cal.cl_ng.shape
    assert model.diagnostics()["cl_x"]["ok"]


def test_mock_model_validate_returns_histogram_report_without_healpy():
    maps = _toy_maps()
    cal = cm.KappaCalibration.from_maps(
        maps,
        nside=2,
        cl_ng=_toy_cls(),
        n_fit_bins=12,
    )
    model = cm.MockModel.fit(cal, order=2, constrained=False, n_jobs=1, quad_order=2)
    mocks = np.stack([maps, maps + 1e-4])

    report = model.validate(mocks, include_spectra=False)

    assert isinstance(report, MockValidationReport)
    assert report.n_mocks == 2
    assert report.n_bins == 2
    assert report.n_pix == maps.shape[1]
    assert report.finite
    assert report.histogram_l1_by_bin.shape == (2,)
    assert not report.spectra_available


def test_mock_model_rejects_unsupported_v1_order():
    cal = cm.KappaCalibration.from_maps(
        _toy_maps(),
        nside=2,
        cl_ng=_toy_cls(),
        n_fit_bins=12,
    )

    with pytest.raises(ValueError, match="orders 2 and 3"):
        cm.MockModel.fit(cal, order=4)


def test_calibration_rejects_asymmetric_spectra():
    cl_ng = _toy_cls()
    cl_ng[0, 1, 2] *= 0.5

    with pytest.raises(ValueError, match="symmetric"):
        cm.KappaCalibration.from_maps(
            _toy_maps(),
            nside=2,
            cl_ng=cl_ng,
            n_fit_bins=12,
        )


def test_calibration_rejects_non_positive_semidefinite_spectra():
    cl_ng = _toy_cls()
    cl_ng[0, 1, 2] = 10.0 * np.sqrt(cl_ng[0, 0, 2] * cl_ng[1, 1, 2])
    cl_ng[1, 0, 2] = cl_ng[0, 1, 2]

    with pytest.raises(ValueError, match="positive semidefinite"):
        cm.KappaCalibration.from_maps(
            _toy_maps(),
            nside=2,
            cl_ng=cl_ng,
            n_fit_bins=12,
        )


def test_missing_healpy_error_for_implicit_spectrum_estimation(monkeypatch):
    import cosmock.calibration as calibration_mod

    def fail_healpy(purpose):
        raise OptionalDependencyError(
            f"{purpose} requires healpy. Install it with `pip install cosmock[healpix]`."
        )

    monkeypatch.setattr(calibration_mod, "require_healpy", fail_healpy)

    with pytest.raises(OptionalDependencyError, match="cosmock\\[healpix\\]"):
        cm.KappaCalibration.from_maps(_toy_maps(n_bins=1), nside=2, n_fit_bins=12)


def test_mock_model_sample_seeded_generation_when_healpy_is_available():
    pytest.importorskip("healpy")

    nside = 1
    lmax = 2
    cl_x = np.zeros((1, 1, lmax + 1), dtype=float)
    cl_x[0, 0] = [1e-20, 1e-20, 1e-3]
    transform_params = np.array([[0.2, 1.0]])
    transform_set = GPTGTransformSet("2", transform_params)
    model = cm.MockModel(
        transform="gptg",
        order="2",
        transform_params=transform_params,
        cl_ng=cl_x,
        cl_x=cl_x,
        nside=nside,
        transform_set=transform_set,
    )

    mock_1 = model.sample(n_mocks=2, seed=123)
    mock_2 = model.sample(n_mocks=2, seed=123)

    assert mock_1.shape == (2, 1, 12)
    assert np.all(np.isfinite(mock_1))
    np.testing.assert_allclose(mock_1, mock_2)
