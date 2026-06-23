"""Mock kappa-map generation from latent Gaussian spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._optional import OptionalDependencyError, require_healpy
from .fitting import fit_transform
from .spectra import target_cls_to_latent_cls
from .transforms import Gn
from .validation import (
    MockValidationReport,
    spectrum_diagnostics,
    validate_cls,
    validate_pixwin,
)


_SUPPORTED_MODEL_ORDERS = {"2", "3"}


def _rng(seed=None, rng=None):
    if rng is not None:
        return rng
    return np.random.default_rng(seed)


def _validate_model_order(order) -> str:
    order = str(order)
    if order not in _SUPPORTED_MODEL_ORDERS:
        raise ValueError("MockModel.fit supports GPTG orders 2 and 3 in v1.")
    return order


def eigvec_matmul(A, x, nbins):
    """Multiply per-ell Cholesky factors into latent alm draws."""

    y = np.zeros_like(x)
    for i in range(nbins):
        for j in range(nbins):
            y[i] += A[i, j] * x[j]
    return y


def apply_cl(xlm, cl, gen_lmax, nbins):
    """Apply target latent spectra to unit Gaussian alm draws."""

    hp = require_healpy("Applying latent spectra to HEALPix alms")
    ell, emm = hp.Alm.getlm(gen_lmax)
    L = np.linalg.cholesky(cl.T).T

    xlm_real = xlm.real
    xlm_imag = xlm.imag
    L_arr = np.swapaxes(L[:, :, ell[ell > -1]], 0, 1)

    ylm_real = eigvec_matmul(L_arr, xlm_real, nbins) / np.sqrt(2.0)
    ylm_imag = eigvec_matmul(L_arr, xlm_imag, nbins) / np.sqrt(2.0)
    ylm_real[:, ell[emm == 0]] *= np.sqrt(2)
    return ylm_real + 1j * ylm_imag


def get_xlm(xlm_real, xlm_imag, gen_lmax, nbins):
    """Build complex HEALPix alms from real and imaginary standard-normal draws."""

    hp = require_healpy("Creating HEALPix alms")
    ell, emm = hp.Alm.getlm(gen_lmax)
    _xlm_real = np.zeros((nbins, len(ell)))
    _xlm_imag = np.zeros_like(_xlm_real)
    _xlm_real[:, ell > 1] = xlm_real
    _xlm_imag[:, (ell > 1) & (emm > 0)] = xlm_imag
    return _xlm_real + 1j * _xlm_imag


def generate_xlm(nbins, gen_lmax, *, seed=None, rng=None):
    """Draw unit Gaussian alms."""

    hp = require_healpy("Drawing HEALPix alms")
    rng = _rng(seed=seed, rng=rng)
    ell, emm = hp.Alm.getlm(gen_lmax)
    xlm_real = rng.normal(size=(nbins, (ell > 1).sum()))
    xlm_imag = rng.normal(size=(nbins, ((ell > 1) & (emm > 0)).sum()))
    xlm = get_xlm(xlm_real, xlm_imag, gen_lmax, nbins)
    return xlm, [xlm_real, xlm_imag]


def generate_mock_y_lm(cl, nbins, gen_lmax, xlms=None, *, seed=None, rng=None):
    """Generate latent Gaussian alms with target spectra."""

    if xlms is not None:
        xlm = xlms
        _xlm = None
    else:
        xlm, _xlm = generate_xlm(nbins, gen_lmax, seed=seed, rng=rng)
    return apply_cl(xlm, cl, gen_lmax, nbins), _xlm


def get_y_maps(cl, nside, nbins, gen_lmax, xlms=None, *, seed=None, rng=None):
    """Generate latent Gaussian HEALPix maps."""

    hp = require_healpy("Generating latent HEALPix maps")
    y_lm, xlm = generate_mock_y_lm(cl, nbins, gen_lmax, xlms, seed=seed, rng=rng)
    y_maps = []
    for i in range(nbins):
        y_map = hp.alm2map(np.ascontiguousarray(y_lm[i]), nside, lmax=gen_lmax, pol=False)
        y_maps.append(y_map)
    return np.array(y_maps), xlm


def get_kappa(y_maps, nbins, N, fitted_params):
    """Apply the fitted transform to latent Gaussian maps."""

    k_list = []
    for i in range(nbins):
        k_nf = Gn(y_maps[i], N, fitted_params[i])
        k_list.append(k_nf)
    return np.array(k_list)


def get_kappa_pixwin(y_maps, nbins, N, fitted_params, nside, pixwinatell):
    """Apply the fitted transform and then a pixel window in harmonic space."""

    hp = require_healpy("Applying a HEALPix pixel window")
    k_list = []
    lmax = 2 * nside
    for i in range(nbins):
        k_nf = Gn(y_maps[i], N, fitted_params[i])
        klm = hp.map2alm(k_nf, lmax=lmax)
        klm = klm * pixwinatell
        k_list.append(hp.alm2map(klm, nside))
    return np.array(k_list)


def get_kappa_lm_pixwin(y_maps, nbins, N, fitted_params, nside, pixwinatell):
    """Return pixel-windowed kappa alms."""

    hp = require_healpy("Applying a HEALPix pixel window")
    k_lm_list = []
    lmax = 2 * nside
    for i in range(nbins):
        k_nf = Gn(y_maps[i], N, fitted_params[i])
        klm = hp.map2alm(k_nf, lmax=lmax)
        klm = klm * pixwinatell
        k_lm_list.append(klm)
    return np.array(k_lm_list)


def _pixwin_at_alm(pixwin, *, nside):
    hp = require_healpy("Applying a HEALPix pixel window")
    lmax = 2 * int(nside)
    pixwin = validate_pixwin(pixwin, lmax=lmax)
    ell_pixwin, _ = hp.Alm.getlm(lmax)
    return pixwin[ell_pixwin]


def create_mock(
    cl_x,
    transform_params,
    *,
    nside,
    order=3,
    seed=None,
    rng=None,
    apply_pixwin=False,
    pixwin=None,
):
    """Create one mock kappa map from latent spectra and transform parameters."""

    params = np.asarray(transform_params, dtype=float)
    if params.ndim != 2:
        raise ValueError("transform_params must have shape (n_bins, n_params).")
    nbins = params.shape[0]
    cl_x = validate_cls(cl_x, n_bins=nbins, name="cl_x")
    if nside is None:
        raise ValueError("nside is required to create HEALPix mock maps.")

    rng = _rng(seed=seed, rng=rng)
    gen_lmax = cl_x.shape[-1] - 1
    y_maps, _ = get_y_maps(cl_x, int(nside), nbins, gen_lmax, rng=rng)

    if apply_pixwin:
        if pixwin is None:
            raise ValueError("pixwin is required when apply_pixwin=True.")
        pixwinatell = _pixwin_at_alm(pixwin, nside=int(nside))
        return get_kappa_pixwin(y_maps, nbins, str(order), params, int(nside), pixwinatell)

    return get_kappa(y_maps, nbins, str(order), params)


@dataclass
class MockModel:
    """Fitted kappa mock generator."""

    transform: str
    order: str
    transform_params: np.ndarray
    cl_ng: np.ndarray
    cl_x: np.ndarray
    nside: int | None
    pixwin: np.ndarray | None = None
    calibration: object | None = None
    constrained: bool = True
    fit_settings: dict[str, object] | None = None

    @classmethod
    def fit(
        cls,
        calibration,
        *,
        transform: str = "gptg",
        order=3,
        constrained: bool = True,
        n_jobs: int = 4,
        **spectra_kwargs,
    ) -> "MockModel":
        """Fit a mock model from a kappa calibration object."""

        if transform != "gptg":
            raise ValueError("Only transform='gptg' is supported in v1.")
        order = _validate_model_order(order)
        if calibration.cl_ng is None:
            raise ValueError("calibration.cl_ng is required to fit latent spectra.")

        params = fit_transform(calibration, order=order, constrained=constrained)
        cl_x = target_cls_to_latent_cls(
            calibration.cl_ng, params, order=order, n_jobs=n_jobs, **spectra_kwargs
        )
        latent_diag = spectrum_diagnostics(cl_x, n_bins=calibration.n_bins, name="cl_x")
        if not latent_diag.ok:
            raise ValueError(
                "Latent spectra are not numerically valid; "
                f"min eigenvalue={latent_diag.min_eigenvalue:.6e}, "
                f"problematic ell values start with {latent_diag.problematic_ells[:5]}."
            )

        return cls(
            transform=transform,
            order=order,
            transform_params=params,
            cl_ng=calibration.cl_ng,
            cl_x=cl_x,
            nside=calibration.nside,
            pixwin=calibration.pixwin,
            calibration=calibration,
            constrained=bool(constrained),
            fit_settings={
                "transform": transform,
                "order": order,
                "constrained": bool(constrained),
                "n_jobs": int(n_jobs),
                "spectra_kwargs": dict(spectra_kwargs),
            },
        )

    @property
    def n_bins(self) -> int:
        """Number of tomographic bins."""

        return int(self.transform_params.shape[0])

    @property
    def n_pix(self) -> int | None:
        """Number of pixels per map when known."""

        if self.calibration is not None:
            return int(self.calibration.n_pix)
        if self.nside is not None:
            return 12 * int(self.nside) * int(self.nside)
        return None

    @property
    def lmax(self) -> int:
        """Maximum ell represented by the model spectra."""

        return int(self.cl_x.shape[-1] - 1)

    def sample(self, *, n_mocks: int = 1, seed=None, apply_pixwin: bool | None = None):
        """Generate mock kappa maps with shape ``(n_mocks, n_bins, n_pix)``."""

        if apply_pixwin is None:
            apply_pixwin = self.pixwin is not None
        return generate_mocks(
            self,
            n_mocks=n_mocks,
            seed=seed,
            apply_pixwin=apply_pixwin,
        )

    def diagnostics(self) -> dict[str, object]:
        """Return model-level numerical diagnostics."""

        return {
            "fit_settings": dict(self.fit_settings or {}),
            "cl_ng": spectrum_diagnostics(
                self.cl_ng, n_bins=self.n_bins, name="cl_ng"
            ).as_dict(),
            "cl_x": spectrum_diagnostics(self.cl_x, n_bins=self.n_bins, name="cl_x").as_dict(),
        }

    def validate(
        self,
        mocks,
        *,
        histogram_bins: int = 50,
        include_spectra: bool | None = None,
        lmax: int | None = None,
        ratio_epsilon: float = 1e-30,
    ) -> MockValidationReport:
        """Compare generated mocks against calibration histograms and spectra."""

        mocks = np.asarray(mocks, dtype=float)
        if mocks.ndim == 2:
            mocks = mocks[np.newaxis, :, :]
        if mocks.ndim != 3:
            raise ValueError("mocks must have shape (n_mocks, n_bins, n_pix).")
        if mocks.shape[1] != self.n_bins:
            raise ValueError(f"mocks has {mocks.shape[1]} bins, expected {self.n_bins}.")
        if self.n_pix is not None and mocks.shape[2] != self.n_pix:
            raise ValueError(f"mocks has {mocks.shape[2]} pixels, expected {self.n_pix}.")

        finite = bool(np.all(np.isfinite(mocks)))
        mean_by_bin = np.mean(mocks, axis=(0, 2))
        std_by_bin = np.std(mocks, axis=(0, 2))

        target_mean = None
        target_std = None
        histogram_l1 = None
        if self.calibration is not None:
            target_maps = np.asarray(self.calibration.maps, dtype=float)
            target_mean = np.mean(target_maps, axis=1)
            target_std = np.std(target_maps, axis=1)
            histogram_l1 = np.zeros(self.n_bins, dtype=float)
            for i in range(self.n_bins):
                target = target_maps[i]
                mock_values = mocks[:, i, :].reshape(-1)
                lo = float(min(np.min(target), np.min(mock_values)))
                hi = float(max(np.max(target), np.max(mock_values)))
                if lo == hi:
                    histogram_l1[i] = 0.0
                    continue
                hist_target, edges = np.histogram(
                    target, bins=histogram_bins, range=(lo, hi), density=True
                )
                hist_mock, _ = np.histogram(
                    mock_values, bins=edges, density=True
                )
                histogram_l1[i] = float(
                    np.sum(np.abs(hist_mock - hist_target) * np.diff(edges))
                )

        spectra_ratio_mean = None
        spectra_ratio_std = None
        hp = None
        if include_spectra is True:
            hp = require_healpy("Validating mock spectra")
        elif include_spectra is None:
            try:
                hp = require_healpy("Validating mock spectra")
            except OptionalDependencyError:
                hp = None

        if hp is not None and self.nside is not None and self.cl_ng is not None:
            lmax_eval = self.lmax if lmax is None else int(lmax)
            lmax_eval = min(lmax_eval, self.cl_ng.shape[-1] - 1, 3 * int(self.nside) - 1)
            cl_samples = np.zeros((mocks.shape[0], self.n_bins, self.n_bins, lmax_eval + 1))
            for mock_idx in range(mocks.shape[0]):
                for i in range(self.n_bins):
                    for j in range(i + 1):
                        cl_ij = hp.anafast(mocks[mock_idx, i], mocks[mock_idx, j], lmax=lmax_eval)
                        cl_samples[mock_idx, i, j] = cl_ij
                        cl_samples[mock_idx, j, i] = cl_ij

            target = self.cl_ng[:, :, : lmax_eval + 1]
            with np.errstate(divide="ignore", invalid="ignore"):
                ratios = np.where(np.abs(target) > ratio_epsilon, cl_samples / target, np.nan)
            spectra_ratio_mean = np.nanmean(ratios, axis=0)
            spectra_ratio_std = np.nanstd(ratios, axis=0)
        elif include_spectra is True:
            raise ValueError("Spectra validation requires model.nside and model.cl_ng.")

        return MockValidationReport(
            n_mocks=int(mocks.shape[0]),
            n_bins=int(mocks.shape[1]),
            n_pix=int(mocks.shape[2]),
            finite=finite,
            mean_by_bin=mean_by_bin,
            target_mean_by_bin=target_mean,
            std_by_bin=std_by_bin,
            target_std_by_bin=target_std,
            histogram_l1_by_bin=histogram_l1,
            spectra_ratio_mean=spectra_ratio_mean,
            spectra_ratio_std=spectra_ratio_std,
        )


def fit_gptg(calibration, *, order=3, **kwargs) -> MockModel:
    """Fit the default GPTG mock model from calibration data."""

    return MockModel.fit(calibration, transform="gptg", order=order, **kwargs)


def generate_mocks(model: MockModel, *, n_mocks: int = 1, seed=None, apply_pixwin=None):
    """Generate one or more mock kappa fields from a fitted model."""

    if n_mocks < 1:
        raise ValueError("n_mocks must be at least 1.")
    if apply_pixwin is None:
        apply_pixwin = model.pixwin is not None
    rng = np.random.default_rng(seed)
    mocks = []
    for _ in range(int(n_mocks)):
        mock = create_mock(
            model.cl_x,
            model.transform_params,
            nside=model.nside,
            order=model.order,
            rng=rng,
            apply_pixwin=apply_pixwin,
            pixwin=model.pixwin,
        )
        mocks.append(mock)
    return np.asarray(mocks)
