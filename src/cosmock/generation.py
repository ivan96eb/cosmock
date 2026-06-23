"""Mock kappa-map generation from latent Gaussian spectra."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transforms import GPTGTransformSet, gptg_transform
from .util.optional import OptionalDependencyError, require_healpy
from .util.validation import (
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


def _spectra_square_root_by_ell(cl, *, psd_atol: float = 1e-12):
    """Return per-ell square-root covariance factors with shape ``(ell, i, j)``."""

    cl_by_ell = np.moveaxis(np.asarray(cl, dtype=float), 2, 0)
    cl_by_ell = 0.5 * (cl_by_ell + np.swapaxes(cl_by_ell, -1, -2))

    try:
        return np.linalg.cholesky(cl_by_ell)
    except np.linalg.LinAlgError as exc:
        eigvals, eigvecs = np.linalg.eigh(cl_by_ell)
        min_eig_by_ell = np.min(eigvals, axis=1)
        bad_ells = np.flatnonzero(min_eig_by_ell < -psd_atol)
        if bad_ells.size:
            first = tuple(int(ell) for ell in bad_ells[:5])
            raise ValueError(
                "cl must be positive semidefinite at each ell before sampling; "
                f"min eigenvalue={float(min_eig_by_ell.min()):.6e}, "
                f"problematic ell values start with {first}."
            ) from exc

        clipped = np.clip(eigvals, 0.0, None)
        return eigvecs * np.sqrt(clipped)[:, np.newaxis, :]


def _apply_cl(xlm, cl, gen_lmax, *, factors_by_ell=None):
    """Apply target latent spectra to unit Gaussian alm draws."""

    hp = require_healpy("Applying latent spectra to HEALPix alms")
    xlm = np.asarray(xlm, dtype=complex)
    cl = np.asarray(cl, dtype=float)
    ell, emm = hp.Alm.getlm(gen_lmax)

    if xlm.ndim != 2:
        raise ValueError("xlm must have shape (n_bins, n_alm).")
    nbins = xlm.shape[0]
    if xlm.shape[1] != len(ell):
        raise ValueError(f"xlm has {xlm.shape[1]} modes, expected {len(ell)}.")
    if cl.ndim != 3 or cl.shape[:2] != (nbins, nbins) or cl.shape[2] <= gen_lmax:
        raise ValueError(
            "cl must have shape (n_bins, n_bins, gen_lmax + 1) for the supplied xlm."
        )

    if factors_by_ell is None:
        factors_by_ell = _spectra_square_root_by_ell(cl)
    else:
        factors_by_ell = np.asarray(factors_by_ell, dtype=float)
        if factors_by_ell.ndim != 3 or factors_by_ell.shape[1:] != (nbins, nbins):
            raise ValueError("factors_by_ell must have shape (lmax + 1, n_bins, n_bins).")
        if factors_by_ell.shape[0] <= gen_lmax:
            raise ValueError("factors_by_ell must include entries through gen_lmax.")
    factors_at_alm = factors_by_ell[ell]

    ylm_real = np.einsum("aij,ja->ia", factors_at_alm, xlm.real) / np.sqrt(2.0)
    ylm_imag = np.einsum("aij,ja->ia", factors_at_alm, xlm.imag) / np.sqrt(2.0)
    ylm_real = np.where(emm == 0, ylm_real * np.sqrt(2.0), ylm_real)
    ylm_imag = np.where(emm == 0, 0.0, ylm_imag)
    return ylm_real + 1j * ylm_imag


def _get_xlm(xlm_real, xlm_imag, gen_lmax, nbins):
    """Build complex HEALPix alms from real and imaginary standard-normal draws."""

    hp = require_healpy("Creating HEALPix alms")
    ell, emm = hp.Alm.getlm(gen_lmax)
    _xlm_real = np.zeros((nbins, len(ell)))
    _xlm_imag = np.zeros_like(_xlm_real)
    _xlm_real[:, ell > 1] = xlm_real
    _xlm_imag[:, (ell > 1) & (emm > 0)] = xlm_imag
    return _xlm_real + 1j * _xlm_imag


def _generate_xlm(nbins, gen_lmax, *, seed=None, rng=None):
    """Draw unit Gaussian alms."""

    hp = require_healpy("Drawing HEALPix alms")
    rng = _rng(seed=seed, rng=rng)
    ell, emm = hp.Alm.getlm(gen_lmax)
    xlm_real = rng.normal(size=(nbins, (ell > 1).sum()))
    xlm_imag = rng.normal(size=(nbins, ((ell > 1) & (emm > 0)).sum()))
    xlm = _get_xlm(xlm_real, xlm_imag, gen_lmax, nbins)
    return xlm, [xlm_real, xlm_imag]


def _generate_mock_y_lm(
    cl,
    nbins,
    gen_lmax,
    xlms=None,
    *,
    seed=None,
    rng=None,
    factors_by_ell=None,
):
    """Generate latent Gaussian alms with target spectra."""

    if xlms is not None:
        xlm = xlms
        _xlm = None
    else:
        xlm, _xlm = _generate_xlm(nbins, gen_lmax, seed=seed, rng=rng)
    return _apply_cl(xlm, cl, gen_lmax, factors_by_ell=factors_by_ell), _xlm


def _get_y_maps(
    cl,
    nside,
    nbins,
    gen_lmax,
    xlms=None,
    *,
    seed=None,
    rng=None,
    factors_by_ell=None,
):
    """Generate latent Gaussian HEALPix maps."""

    hp = require_healpy("Generating latent HEALPix maps")
    y_lm, xlm = _generate_mock_y_lm(
        cl,
        nbins,
        gen_lmax,
        xlms,
        seed=seed,
        rng=rng,
        factors_by_ell=factors_by_ell,
    )
    y_maps = []
    for i in range(nbins):
        y_map = hp.alm2map(np.ascontiguousarray(y_lm[i]), nside, lmax=gen_lmax, pol=False)
        y_maps.append(y_map)
    return np.array(y_maps), xlm


def _evaluate_gptg_by_bin(y_maps, nbins, order, fitted_params):
    """Evaluate GPTG transforms for all bins with one broadcasted NumPy expression."""

    order = str(order)
    y_maps = np.asarray(y_maps, dtype=float)
    params = np.asarray(fitted_params, dtype=float)
    if y_maps.ndim != 2 or y_maps.shape[0] != nbins:
        raise ValueError("y_maps must have shape (n_bins, n_pix).")
    if params.shape[0] != nbins:
        raise ValueError("fitted_params must contain one parameter row per bin.")

    if order == "2":
        if params.shape[1] != 2:
            raise ValueError("G2 expects two fitted parameters per bin.")
        alpha = params[:, 0, np.newaxis]
        beta = params[:, 1, np.newaxis]
        return beta * np.exp(alpha * y_maps - 0.5 * alpha**2) - beta

    if order == "3":
        if params.shape[1] != 3:
            raise ValueError("G3 expects three fitted parameters per bin.")
        a = params[:, 0, np.newaxis]
        b = params[:, 1, np.newaxis]
        c = params[:, 2, np.newaxis]
        arg = np.exp(a * y_maps - 0.5 * a**2) + b * y_maps + c
        return arg / (1.0 + c) - 1.0

    return np.array([gptg_transform(y_maps[i], order, params[i]) for i in range(nbins)])


def _get_kappa(y_maps, nbins, N, fitted_params):
    """Apply the fitted transform to latent Gaussian maps."""

    return _evaluate_gptg_by_bin(y_maps, nbins, N, fitted_params)


def _get_kappa_pixwin(y_maps, nbins, N, fitted_params, nside, pixwinatell):
    """Apply the fitted transform and then a pixel window in harmonic space."""

    hp = require_healpy("Applying a HEALPix pixel window")
    k_list = []
    lmax = 2 * nside
    ell, _ = hp.Alm.getlm(lmax)
    pixwinatell = np.asarray(pixwinatell, dtype=float)
    if pixwinatell.shape != (len(ell),):
        raise ValueError("pixwinatell must have one value per alm mode for lmax=2*nside.")
    kappa_maps = _evaluate_gptg_by_bin(y_maps, nbins, N, fitted_params)
    for i in range(nbins):
        k_nf = kappa_maps[i]
        klm = hp.map2alm(k_nf, lmax=lmax)
        klm = klm * pixwinatell
        k_list.append(hp.alm2map(klm, nside, lmax=lmax))
    return np.array(k_list)


def _get_kappa_lm_pixwin(y_maps, nbins, N, fitted_params, nside, pixwinatell):
    """Return pixel-windowed kappa alms."""

    hp = require_healpy("Applying a HEALPix pixel window")
    k_lm_list = []
    lmax = 2 * nside
    ell, _ = hp.Alm.getlm(lmax)
    pixwinatell = np.asarray(pixwinatell, dtype=float)
    if pixwinatell.shape != (len(ell),):
        raise ValueError("pixwinatell must have one value per alm mode for lmax=2*nside.")
    kappa_maps = _evaluate_gptg_by_bin(y_maps, nbins, N, fitted_params)
    for i in range(nbins):
        k_nf = kappa_maps[i]
        klm = hp.map2alm(k_nf, lmax=lmax)
        klm = klm * pixwinatell
        k_lm_list.append(klm)
    return np.array(k_lm_list)


def _pixwin_at_alm(pixwin, *, nside, lmax=None):
    """Expand an ell-indexed pixel window to one factor per HEALPix alm mode."""

    hp = require_healpy("Applying a HEALPix pixel window")
    lmax = 2 * int(nside) if lmax is None else int(lmax)
    pixwin = validate_pixwin(pixwin, lmax=lmax)
    ell_pixwin, _ = hp.Alm.getlm(lmax)
    return pixwin[ell_pixwin]


def _create_mock(
    cl_x,
    transform_params,
    *,
    nside,
    order=3,
    seed=None,
    rng=None,
    apply_pixwin=False,
    pixwin=None,
    factors_by_ell=None,
    pixwinatell=None,
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
    y_maps, _ = _get_y_maps(
        cl_x,
        int(nside),
        nbins,
        gen_lmax,
        rng=rng,
        factors_by_ell=factors_by_ell,
    )

    if apply_pixwin:
        if pixwin is None and pixwinatell is None:
            raise ValueError("pixwin is required when apply_pixwin=True.")
        if pixwinatell is None:
            pixwinatell = _pixwin_at_alm(pixwin, nside=int(nside))
        return _get_kappa_pixwin(y_maps, nbins, str(order), params, int(nside), pixwinatell)

    return _get_kappa(y_maps, nbins, str(order), params)


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
    transform_set: GPTGTransformSet | None = None
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
        initial_params=None,
        **spectra_kwargs,
    ) -> "MockModel":
        """Fit a mock model from a kappa calibration object."""

        if transform != "gptg":
            raise ValueError("Only transform='gptg' is supported in v1.")
        order = _validate_model_order(order)
        if calibration.cl_ng is None:
            raise ValueError("calibration.cl_ng is required to fit latent spectra.")

        transform_set = calibration.fit_transform(
            order=order,
            constrained=constrained,
            initial_params=initial_params,
        )
        params = transform_set.transform_params
        cl_x = transform_set.to_latent_spectra(n_jobs=n_jobs, **spectra_kwargs)
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
            transform_set=transform_set,
            constrained=bool(constrained),
            fit_settings={
                "transform": transform,
                "order": order,
                "constrained": bool(constrained),
                "n_jobs": int(n_jobs),
                "initial_params": initial_params is not None,
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
        return _generate_mocks(
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


def _generate_mocks(model: MockModel, *, n_mocks: int = 1, seed=None, apply_pixwin=None):
    """Generate one or more mock kappa fields from a fitted model."""

    if n_mocks < 1:
        raise ValueError("n_mocks must be at least 1.")
    if apply_pixwin is None:
        apply_pixwin = model.pixwin is not None
    rng = np.random.default_rng(seed)
    cl_x = validate_cls(model.cl_x, n_bins=model.n_bins, name="cl_x")
    factors_by_ell = _spectra_square_root_by_ell(cl_x)
    pixwinatell = None
    if apply_pixwin:
        if model.pixwin is None:
            raise ValueError("model.pixwin is required when apply_pixwin=True.")
        if model.nside is None:
            raise ValueError("model.nside is required when apply_pixwin=True.")
        pixwinatell = _pixwin_at_alm(model.pixwin, nside=int(model.nside))
    mocks = []
    for _ in range(int(n_mocks)):
        mock = _create_mock(
            cl_x,
            model.transform_params,
            nside=model.nside,
            order=model.order,
            rng=rng,
            apply_pixwin=apply_pixwin,
            pixwin=model.pixwin,
            factors_by_ell=factors_by_ell,
            pixwinatell=pixwinatell,
        )
        mocks.append(mock)
    return np.asarray(mocks)
