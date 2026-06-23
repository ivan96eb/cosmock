"""Calibration objects and empirical map statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .util.optional import require_healpy
from .util.statistics import histogramer2d as _histogramer2d
from .util.validation import validate_cls, validate_kappa_maps, validate_pixwin


def estimate_cls_from_maps(kappa_maps, *, nside: int | None = None, lmax: int | None = None):
    """Estimate auto/cross spectra from kappa maps using healpy."""

    hp = require_healpy("Estimating spectra from maps")
    maps = validate_kappa_maps(kappa_maps, nside=nside)
    if nside is None:
        nside = hp.npix2nside(maps.shape[1])
    if lmax is None:
        lmax = 3 * int(nside) - 1

    n_bins = maps.shape[0]
    cl_ng = np.zeros((n_bins, n_bins, lmax + 1), dtype=float)
    for i in range(n_bins):
        for j in range(i + 1):
            cl_ij = hp.anafast(maps[i], maps[j], lmax=lmax)
            cl_ng[i, j] = cl_ij
            cl_ng[j, i] = cl_ij
    return cl_ng


@dataclass
class KappaCalibration:
    """Calibration data derived from one or more input kappa maps."""

    maps: np.ndarray
    nside: int | None
    n_bins: int
    n_pix: int
    gaussianized_samples: list[np.ndarray]
    binned_x: list[np.ndarray]
    binned_y: list[np.ndarray]
    cl_ng: np.ndarray | None = None
    pixwin: np.ndarray | None = None
    x_range: tuple[float, float] = (-10.0, 4.5)
    n_fit_bins: int = 1000

    @classmethod
    def from_maps(
        cls,
        kappa_maps,
        *,
        nside: int | None = None,
        cl_ng=None,
        pixwin=None,
        x_range=(-10.0, 4.5),
        n_fit_bins: int = 1000,
        lmax: int | None = None,
    ) -> "KappaCalibration":
        """Build calibration statistics from input kappa maps."""

        maps = validate_kappa_maps(kappa_maps, nside=nside)
        n_bins, n_pix = maps.shape

        if cl_ng is None:
            cl_ng_arr = estimate_cls_from_maps(maps, nside=nside, lmax=lmax)
        else:
            cl_ng_arr = validate_cls(cl_ng, n_bins=n_bins, name="cl_ng")

        pixwin_arr = None
        if pixwin is not None:
            required_lmax = cl_ng_arr.shape[-1] - 1 if cl_ng_arr is not None else lmax
            pixwin_arr = validate_pixwin(pixwin, lmax=required_lmax)

        gaussianized_samples: list[np.ndarray] = []
        binned_x: list[np.ndarray] = []
        binned_y: list[np.ndarray] = []
        for map_values in maps:
            x_gaussianized, x_avg, y_avg = _histogramer2d(
                map_values, n_fit_bins, x_range=x_range
            )
            gaussianized_samples.append(x_gaussianized)
            binned_x.append(x_avg)
            binned_y.append(y_avg)

        return cls(
            maps=maps,
            nside=nside,
            n_bins=n_bins,
            n_pix=n_pix,
            gaussianized_samples=gaussianized_samples,
            binned_x=binned_x,
            binned_y=binned_y,
            cl_ng=cl_ng_arr,
            pixwin=pixwin_arr,
            x_range=tuple(x_range),
            n_fit_bins=int(n_fit_bins),
        )

    def fit_transform(self, *, order=3, constrained: bool = True, initial_params=None):
        """Fit GPTG transform parameters for this calibration."""

        from .fitting import fit_transform as _fit_transform
        from .transforms import GPTGTransformSet

        params = _fit_transform(
            self,
            order=order,
            constrained=constrained,
            initial_params=initial_params,
        )
        return GPTGTransformSet(
            order=str(order),
            params=params,
            calibration=self,
            constrained=bool(constrained),
            fit_settings={
                "order": str(order),
                "constrained": bool(constrained),
                "initial_params": initial_params is not None,
            },
        )
