from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import eval_legendre

import cosmock as cm
from cosmock.fitting import fit_gn, fit_gn_with_constraint, fit_transform, variance_from_Cl
from cosmock.spectra import target_cls_to_latent_cls, var_pdf
from cosmock.transforms import GPTGTransform, GPTGTransformSet, gptg_transform
from cosmock.util.statistics import empirical_cdf, histogramer2d


def _toy_cls_for_variance(variance: float, lmax: int = 7):
    cl = np.zeros(lmax + 1, dtype=float)
    cl[2] = variance * 4 * np.pi / 5
    return cl


def _latent_cls(n_bins=2, lmax=5):
    cl_x = np.zeros((n_bins, n_bins, lmax + 1), dtype=float)
    base = np.zeros(lmax + 1, dtype=float)
    base[2:] = np.array([1e-4, 4e-5, 2e-5, 1e-5])[: lmax - 1]
    for i in range(n_bins):
        cl_x[i, i] = (1.0 + 0.5 * i) * base
    for i in range(n_bins):
        for j in range(i):
            cl_x[i, j] = 0.2 * np.sqrt(cl_x[i, i] * cl_x[j, j])
            cl_x[j, i] = cl_x[i, j]
    cl_x[:, :, 0] = 1e-20 * np.eye(n_bins)
    cl_x[:, :, 1] = 1e-20 * np.eye(n_bins)
    return cl_x


def _g2_forward_target_cls(cl_x, transform_params, *, n_quad=80):
    """Construct target spectra from latent spectra using the analytic G2 relation."""

    n_bins = cl_x.shape[0]
    lmax = cl_x.shape[-1] - 1
    ell = np.arange(lmax + 1)
    mu, weights = np.polynomial.legendre.leggauss(n_quad)
    p_ell = np.array([eval_legendre(l, mu) for l in ell])
    cl_ng = np.zeros_like(cl_x)

    for i in range(n_bins):
        for j in range(n_bins):
            xi_g = np.sum(
                (2 * ell[:, np.newaxis] + 1) * p_ell * cl_x[i, j, :, np.newaxis],
                axis=0,
            ) / (4 * np.pi)
            alpha_i, beta_i = transform_params[i]
            alpha_j, beta_j = transform_params[j]
            xi_ng = beta_i * beta_j * (np.exp(alpha_i * alpha_j * xi_g) - 1)
            cl_ng[i, j] = 2 * np.pi * np.sum(
                weights[np.newaxis, :] * p_ell * xi_ng[np.newaxis, :],
                axis=1,
            )

    cl_ng[:, :, 0] = 1e-20 * np.eye(n_bins)
    cl_ng[:, :, 1] = 1e-20 * np.eye(n_bins)
    return cl_ng


def test_empirical_cdf_and_gaussianized_bins_preserve_rank_state():
    input_map = np.array([-2.0, -0.5, 0.0, 1.25, 3.0, 4.0])

    sorted_map, cdf = empirical_cdf(input_map)
    gaussianized, x_bins, y_bins = histogramer2d(input_map, 4, x_range=(-5, 5))

    np.testing.assert_array_equal(sorted_map, np.sort(input_map))
    assert np.all(np.diff(cdf) > 0)
    assert np.all((cdf > 0) & (cdf < 1))
    assert np.all(np.diff(gaussianized) > 0)
    assert np.all(np.diff(x_bins) > 0)
    assert np.all(np.diff(y_bins) >= 0)
    assert y_bins.min() >= input_map.min()
    assert y_bins.max() <= input_map.max()


@pytest.mark.parametrize(
    ("order", "true_params", "initial_params", "atol"),
    [
        ("2", np.array([0.35, 0.8]), np.array([0.3, 0.7]), 1e-5),
        ("3", np.array([0.25, 0.15, 0.1]), np.array([0.2, 0.1, 0.05]), 1e-3),
    ],
)
def test_unconstrained_fit_recovers_known_transform_parameters(
    order, true_params, initial_params, atol
):
    x_data = np.linspace(-3.0, 3.0, 200)
    y_data = gptg_transform(x_data, order, true_params)

    fitted = fit_gn(x_data, y_data, order, initial_params=initial_params)

    assert fitted.shape == true_params.shape
    np.testing.assert_allclose(fitted, true_params, atol=atol, rtol=0)


@pytest.mark.parametrize(
    ("order", "true_params", "initial_params"),
    [
        ("2", np.array([0.35, 0.8]), np.array([0.75, np.nan])),
        ("3", np.array([0.25, 0.15, 0.1]), np.array([0.2, 0.1, np.nan])),
    ],
)
def test_constrained_fit_matches_variance_encoded_by_target_cls(
    order, true_params, initial_params
):
    x_data = np.linspace(-3.0, 3.0, 240)
    y_data = gptg_transform(x_data, order, true_params)
    target_variance = var_pdf(order, true_params, n_nodes=80)
    cl = _toy_cls_for_variance(target_variance)

    fitted = fit_gn_with_constraint(
        x_data,
        y_data,
        order,
        cl,
        initial_params=initial_params,
    )

    np.testing.assert_allclose(var_pdf(order, fitted, n_nodes=80), variance_from_Cl(cl), rtol=1e-9)


def test_g2_target_to_latent_cls_round_trips_analytic_input_state():
    transform_params = np.array([[0.2, 0.8], [0.25, 1.1]])
    cl_x_input = _latent_cls(n_bins=2, lmax=5)
    cl_ng = _g2_forward_target_cls(cl_x_input, transform_params)

    cl_x_recovered = target_cls_to_latent_cls(
        cl_ng,
        transform_params,
        order=2,
        n_jobs=1,
        quad_order=12,
    )

    np.testing.assert_allclose(cl_x_recovered, cl_x_input, atol=5e-14, rtol=0)


def test_gptg_transform_objects_match_function_kernel():
    x_data = np.linspace(-2.0, 2.0, 30)
    params = np.array([0.3, 0.9])
    transform = GPTGTransform("2", params)
    transform_set = GPTGTransformSet("2", params[np.newaxis, :])

    np.testing.assert_allclose(transform.evaluate(x_data), gptg_transform(x_data, "2", params))
    np.testing.assert_allclose(
        transform_set.evaluate(x_data, bin_index=0),
        gptg_transform(x_data, "2", params),
    )


@pytest.mark.parametrize("constrained", [False, True])
def test_high_level_and_three_function_spine_match_for_pipeline_state(constrained):
    maps = np.vstack(
        [
            gptg_transform(np.linspace(-2.5, 2.5, 64), "2", [0.2, 0.6]),
            gptg_transform(np.linspace(-2.5, 2.5, 64), "2", [0.25, 0.7]),
        ]
    )
    cl_ng = _g2_forward_target_cls(
        _latent_cls(n_bins=2, lmax=5),
        np.array([[0.2, 0.6], [0.25, 0.7]]),
    )
    cal = cm.KappaCalibration.from_maps(
        maps,
        cl_ng=cl_ng,
        n_fit_bins=24,
    )

    transform_params = fit_transform(cal, order=2, constrained=constrained)
    transform_set = cal.fit_transform(order=2, constrained=constrained)
    cl_x = target_cls_to_latent_cls(
        cal.cl_ng,
        transform_params,
        order=2,
        n_jobs=1,
        quad_order=12,
    )
    cl_x_from_object = transform_set.to_latent_spectra(n_jobs=1, quad_order=12)
    model = cm.MockModel.fit(
        cal,
        order=2,
        constrained=constrained,
        n_jobs=1,
        quad_order=12,
    )

    np.testing.assert_allclose(transform_set.transform_params, transform_params)
    np.testing.assert_allclose(cl_x_from_object, cl_x)
    np.testing.assert_allclose(model.transform_params, transform_params)
    assert model.transform_set is not None
    np.testing.assert_allclose(model.transform_set.transform_params, transform_params)
    np.testing.assert_allclose(model.cl_x, cl_x)
    np.testing.assert_allclose(model.cl_ng, cal.cl_ng)


def test_core_math_matches_legacy_notebook_functions_when_available():
    legacy_root = Path("/home/denniswu28/Extended_Lognormal")
    if not legacy_root.exists():
        pytest.skip("legacy Extended_Lognormal repository is not available")

    sys.path.insert(0, str(legacy_root))
    try:
        legacy_gn = importlib.import_module("fitter.Gn").Gn
        legacy_histogramer2d = importlib.import_module("fitter.auxiliary_functions").histogramer2d
        legacy_fit_gn = importlib.import_module("fitter.fitter").fit_gn

        x_data = np.linspace(-2.0, 2.0, 40)
        params = np.array([0.2, 0.8])
        input_map = np.linspace(-1.0, 1.0, 40) ** 3

        np.testing.assert_allclose(
            legacy_gn(x_data, "2", params),
            gptg_transform(x_data, "2", params),
        )

        legacy_hist = legacy_histogramer2d(input_map, 8, (-3, 3))
        new_hist = histogramer2d(input_map, 8, (-3, 3))
        for legacy_arr, new_arr in zip(legacy_hist, new_hist):
            np.testing.assert_allclose(legacy_arr, new_arr)

        y_data = gptg_transform(x_data, "2", params)
        initial_params = np.array([0.1, 0.9])
        np.testing.assert_allclose(
            legacy_fit_gn(x_data, y_data, "2", initial_params=initial_params),
            fit_gn(x_data, y_data, "2", initial_params=initial_params),
        )
    finally:
        sys.path.remove(str(legacy_root))
