from __future__ import annotations

import numpy as np
import pytest
from scipy.special import eval_legendre

import cosmock as cm
import cosmock.generation as generation_mod
import cosmock.healpix as healpix_mod
from cosmock.fitting import fit_gn, fit_gn_with_constraint, fit_transform, variance_from_Cl
from cosmock.spectra import (
    build_lookup_table,
    correct_cl,
    target_cls_to_latent_cls,
    var_pdf,
)
from cosmock.transforms import GPTGTransform, GPTGTransformSet, gptg_transform
from cosmock.util.quadrature import get_gh_nodes_weights
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
    p_ell = np.array([eval_legendre(ell_value, mu) for ell_value in ell])
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


def _fake_getlm(lmax):
    ell = []
    emm = []
    for m in range(lmax + 1):
        for ell_value in range(m, lmax + 1):
            ell.append(ell_value)
            emm.append(m)
    return np.asarray(ell), np.asarray(emm)


class _FakeAlm:
    @staticmethod
    def getlm(lmax):
        return _fake_getlm(lmax)


class _FakeHealpy:
    Alm = _FakeAlm

    @staticmethod
    def alm2map(alm, nside, lmax=None, pol=False):
        npix = 12 * int(nside) * int(nside)
        return np.resize(np.asarray(alm).real, npix)

    @staticmethod
    def map2alm(values, lmax):
        nalm = len(_fake_getlm(lmax)[0])
        return np.resize(np.asarray(values, dtype=float), nalm).astype(complex)


def _reference_F_gauss_hermite_single(n, params_i, params_j, xi_g, precomputed):
    y_nodes, y_weights = precomputed
    cov = np.array([[1.0, xi_g], [xi_g, 1.0]])
    chol = np.linalg.cholesky(cov)
    yi, yj = np.meshgrid(y_nodes, y_nodes, indexing="ij")
    wi, wj = np.meshgrid(y_weights, y_weights, indexing="ij")
    ystack = np.stack([yi.ravel(), yj.ravel()], axis=1)
    xstack = (chol @ ystack.T).T
    gn_i = gptg_transform(xstack[:, 0], n, params_i)
    gn_j = gptg_transform(xstack[:, 1], n, params_j)
    return np.sum(gn_i * gn_j * (wi * wj).ravel())


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
    ("order", "params"),
    [
        ("2", np.array([0.35, 0.8])),
        ("3", np.array([0.25, 0.15, 0.1])),
    ],
)
def test_gptg_transform_mean_zero_and_variance_formula(order, params):
    nodes, weights = get_gh_nodes_weights(80)
    transformed = gptg_transform(nodes, order, params)

    if order == "2":
        alpha, beta = params
        expected_variance = beta**2 * (np.exp(alpha**2) - 1.0)
    else:
        a, b, c = params
        expected_variance = (np.exp(a**2) - 1.0 + 2.0 * a * b + b**2) / (1.0 + c) ** 2

    np.testing.assert_allclose(np.sum(weights * transformed), 0.0, atol=1e-13)
    np.testing.assert_allclose(var_pdf(order, params, n_nodes=80), expected_variance, rtol=1e-13)


@pytest.mark.parametrize("order", ["4", "5"])
def test_gptg_rejects_unrestored_v1_orders(order):
    with pytest.raises(ValueError, match="orders 2 and 3"):
        gptg_transform(np.array([0.0]), order, np.ones(int(order)))
    with pytest.raises(ValueError, match="orders 2 and 3"):
        GPTGTransform(order, np.ones(int(order)))


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


def test_g2_latent_conversion_rejects_invalid_log_domain():
    cl_ng = np.zeros((2, 2, 4), dtype=float)
    cl_ng[:, :, 0] = 1e-20 * np.eye(2)
    cl_ng[:, :, 1] = 1e-20 * np.eye(2)
    cl_ng[0, 0, 2] = 1.0
    cl_ng[1, 1, 2] = 1.0
    cl_ng[0, 1, 2] = 0.5
    cl_ng[1, 0, 2] = 0.5
    transform_params = np.array([[0.2, 0.1], [0.2, -0.1]])

    with pytest.raises(ValueError, match="G2 inverse log-domain"):
        target_cls_to_latent_cls(
            cl_ng,
            transform_params,
            order=2,
            n_jobs=1,
            quad_order=12,
        )


def test_g3_lookup_table_matches_reference_and_is_monotone():
    params_i = np.array([0.25, 0.12, 0.08])
    params_j = np.array([0.28, 0.09, 0.05])
    xi_g_grid = np.linspace(-0.8, 0.8, 9)
    precomputed = get_gh_nodes_weights(12)

    lookup = build_lookup_table("3", params_i, params_j, xi_g_grid, precomputed, nnodes=12)
    reference = np.array(
        [
            _reference_F_gauss_hermite_single("3", params_i, params_j, xi_g, precomputed)
            for xi_g in xi_g_grid
        ]
    )

    np.testing.assert_allclose(lookup, reference, rtol=1e-13, atol=1e-15)
    assert np.all(np.diff(lookup) > 0.0)


def test_g3_latent_conversion_rejects_non_monotone_lookup():
    cl_ng = np.zeros((1, 1, 3), dtype=float)
    cl_ng[:, :, 0] = np.eye(1) * 1e-20
    cl_ng[:, :, 1] = np.eye(1) * 1e-20
    transform_params = np.array([[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError, match="monotone"):
        target_cls_to_latent_cls(
            cl_ng,
            transform_params,
            order=3,
            n_jobs=1,
            quad_order=4,
            xig_grid_size=9,
        )


def test_g3_latent_conversion_rejects_lookup_extrapolation():
    cl_ng = np.zeros((1, 1, 3), dtype=float)
    cl_ng[:, :, 0] = np.eye(1) * 1e-20
    cl_ng[:, :, 1] = np.eye(1) * 1e-20
    cl_ng[0, 0, 2] = 10.0
    transform_params = np.array([[0.25, 0.12, 0.08]])

    with pytest.raises(ValueError, match="outside the G3 correlation lookup range"):
        target_cls_to_latent_cls(
            cl_ng,
            transform_params,
            order=3,
            n_jobs=1,
            quad_order=4,
            xig_grid_size=21,
        )


def test_latent_conversion_forces_low_ell_identity_convention():
    transform_params = np.array([[0.2, 0.8]])
    cl_ng = _g2_forward_target_cls(_latent_cls(n_bins=1, lmax=5), transform_params)

    cl_x = target_cls_to_latent_cls(
        cl_ng,
        transform_params,
        order=2,
        n_jobs=1,
        quad_order=12,
    )

    np.testing.assert_allclose(cl_x[:, :, 0], np.eye(1) * 1e-20)
    np.testing.assert_allclose(cl_x[:, :, 1], np.eye(1) * 1e-20)


def test_correct_cl_handles_zero_low_ell_without_nonfinite_values():
    cl = np.zeros((1, 1, 4), dtype=float)
    cl[0, 0, 2:] = [1e-4, 5e-5]
    transform_params = np.array([[0.2, 0.8]])

    corrected = correct_cl(cl, "2", transform_params, 1, A=np.array([1e-5]))

    assert np.all(np.isfinite(corrected))
    np.testing.assert_allclose(corrected[:, :, 0], np.eye(1) * 1e-20)
    np.testing.assert_allclose(corrected[:, :, 1], np.eye(1) * 1e-20)


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


def test_generation_xlm_builder_masks_low_ell_and_monopole_imag(monkeypatch):
    monkeypatch.setattr(generation_mod, "require_healpy", lambda purpose: _FakeHealpy)
    ell, emm = _fake_getlm(3)
    nbins = 2
    xlm_real = np.arange(nbins * np.count_nonzero(ell > 1), dtype=float).reshape(
        nbins, -1
    )
    xlm_imag = np.arange(
        10,
        10 + nbins * np.count_nonzero((ell > 1) & (emm > 0)),
        dtype=float,
    ).reshape(nbins, -1)

    xlm = generation_mod._get_xlm(xlm_real, xlm_imag, 3, nbins)

    assert xlm.shape == (nbins, len(ell))
    np.testing.assert_allclose(xlm.real[:, ell <= 1], 0.0)
    np.testing.assert_allclose(xlm.real[:, ell > 1], xlm_real)
    np.testing.assert_allclose(xlm.imag[:, emm == 0], 0.0)
    np.testing.assert_allclose(xlm.imag[:, (ell > 1) & (emm > 0)], xlm_imag)


def test_generation_apply_cl_matches_per_ell_cholesky_einsum(monkeypatch):
    monkeypatch.setattr(generation_mod, "require_healpy", lambda purpose: _FakeHealpy)
    gen_lmax = 2
    ell, emm = _fake_getlm(gen_lmax)
    cl = np.zeros((2, 2, gen_lmax + 1), dtype=float)
    cl[:, :, 0] = np.array([[1e-20, 0.0], [0.0, 2e-20]])
    cl[:, :, 1] = np.array([[0.4, 0.1], [0.1, 0.3]])
    cl[:, :, 2] = np.array([[1.2, 0.2], [0.2, 0.8]])
    xlm = np.array(
        [
            [1 + 10j, 2 + 20j, 3 + 30j, 4 + 40j, 5 + 50j, 6 + 60j],
            [-1 - 10j, -2 - 20j, -3 - 30j, -4 - 40j, -5 - 50j, -6 - 60j],
        ],
        dtype=complex,
    )
    chol = np.moveaxis(np.linalg.cholesky(np.moveaxis(cl, 2, 0)), 0, 2)
    chol_at_alm = chol[:, :, ell]
    expected_real = np.einsum("ijm,jm->im", chol_at_alm, xlm.real) / np.sqrt(2.0)
    expected_imag = np.einsum("ijm,jm->im", chol_at_alm, xlm.imag) / np.sqrt(2.0)
    expected_real = np.where(emm == 0, expected_real * np.sqrt(2.0), expected_real)
    expected_imag = np.where(emm == 0, 0.0, expected_imag)

    factors = generation_mod._spectra_square_root_by_ell(cl)
    ylm = generation_mod._apply_cl(xlm, cl, gen_lmax)
    ylm_with_factors = generation_mod._apply_cl(
        xlm,
        cl,
        gen_lmax,
        factors_by_ell=factors,
    )

    np.testing.assert_allclose(ylm, expected_real + 1j * expected_imag)
    np.testing.assert_allclose(ylm_with_factors, expected_real + 1j * expected_imag)
    np.testing.assert_allclose(ylm.imag[:, emm == 0], 0.0)


def test_generation_square_root_handles_semidefinite_spectra():
    cl = np.zeros((2, 2, 3), dtype=float)
    cl[:, :, 0] = np.array([[0.0, 0.0], [0.0, 0.0]])
    cl[:, :, 1] = np.array([[1.0, 1.0], [1.0, 1.0]])
    cl[:, :, 2] = np.array([[4.0, 2.0], [2.0, 1.0]])

    factors = generation_mod._spectra_square_root_by_ell(cl)
    reconstructed = np.einsum("lij,lkj->lik", factors, factors)

    np.testing.assert_allclose(reconstructed, np.moveaxis(cl, 2, 0), atol=1e-14)


def test_generation_square_root_rejects_indefinite_spectra():
    cl = np.zeros((2, 2, 2), dtype=float)
    cl[:, :, 0] = np.eye(2)
    cl[:, :, 1] = np.array([[1.0, 2.0], [2.0, 1.0]])

    with pytest.raises(ValueError, match="positive semidefinite"):
        generation_mod._spectra_square_root_by_ell(cl)


def test_mock_model_sample_uses_new_generation_path_with_mocked_healpy(monkeypatch):
    monkeypatch.setattr(generation_mod, "require_healpy", lambda purpose: _FakeHealpy)
    nside = 1
    lmax = 2
    cl_x = np.zeros((1, 1, lmax + 1), dtype=float)
    cl_x[0, 0] = [1e-20, 1e-20, 1e-3]
    transform_params = np.array([[0.2, 1.0]])
    model = cm.MockModel(
        transform="gptg",
        order="2",
        transform_params=transform_params,
        cl_ng=cl_x,
        cl_x=cl_x,
        nside=nside,
        transform_set=GPTGTransformSet("2", transform_params),
    )

    mock_1 = model.sample(n_mocks=2, seed=123)
    mock_2 = model.sample(n_mocks=2, seed=123)

    assert mock_1.shape == (2, 1, 12)
    assert np.all(np.isfinite(mock_1))
    np.testing.assert_allclose(mock_1, mock_2)


def test_mock_model_sample_reuses_precomputed_spectrum_factors(monkeypatch):
    monkeypatch.setattr(generation_mod, "require_healpy", lambda purpose: _FakeHealpy)
    original = generation_mod._spectra_square_root_by_ell
    calls = 0

    def counting_square_root(cl, *, psd_atol=1e-12):
        nonlocal calls
        calls += 1
        return original(cl, psd_atol=psd_atol)

    monkeypatch.setattr(generation_mod, "_spectra_square_root_by_ell", counting_square_root)
    nside = 1
    lmax = 2
    cl_x = np.zeros((1, 1, lmax + 1), dtype=float)
    cl_x[0, 0] = [1e-20, 1e-20, 1e-3]
    transform_params = np.array([[0.2, 1.0]])
    model = cm.MockModel(
        transform="gptg",
        order="2",
        transform_params=transform_params,
        cl_ng=cl_x,
        cl_x=cl_x,
        nside=nside,
        transform_set=GPTGTransformSet("2", transform_params),
    )

    model.sample(n_mocks=3, seed=123)

    assert calls == 1


def test_pixwin_at_alm_expands_ell_window_and_checks_lmax(monkeypatch):
    monkeypatch.setattr(generation_mod, "require_healpy", lambda purpose: _FakeHealpy)
    ell, _ = _fake_getlm(2)
    pixwin = np.array([1.0, 0.8, 0.5])

    pixwinatell = generation_mod._pixwin_at_alm(pixwin, nside=1)

    np.testing.assert_allclose(pixwinatell, pixwin[ell])
    with pytest.raises(ValueError, match="ell=2"):
        generation_mod._pixwin_at_alm(np.ones(2), nside=1)


def test_healpix_pixwin_helper_treats_windows_as_ell_indexed():
    ell = np.array([0, 1, 2, 2, 3])
    pixwin = np.array([1.0, 0.9, 0.75, 0.6])

    np.testing.assert_allclose(
        healpix_mod._ell_indexed_window_values(pixwin, ell),
        np.array([1.0, 0.9, 0.75, 0.75, 0.6]),
    )
    with pytest.raises(ValueError, match="ell=3"):
        healpix_mod._ell_indexed_window_values(np.ones(3), ell)


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
