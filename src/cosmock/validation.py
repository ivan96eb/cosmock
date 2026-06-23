"""Validation helpers and report objects for cosmock inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectrumDiagnostics:
    """Numerical health summary for a tomographic spectra cube."""

    name: str
    shape: tuple[int, ...]
    finite: bool
    symmetric: bool
    min_eigenvalue: float
    problematic_ells: tuple[int, ...]

    @property
    def positive_semidefinite(self) -> bool:
        """Whether every ell covariance matrix is numerically valid."""

        return len(self.problematic_ells) == 0

    @property
    def ok(self) -> bool:
        """Whether the spectra cube is finite, symmetric, and semidefinite."""

        return self.finite and self.symmetric and self.positive_semidefinite

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly diagnostics dictionary."""

        return {
            "name": self.name,
            "shape": self.shape,
            "finite": self.finite,
            "symmetric": self.symmetric,
            "min_eigenvalue": self.min_eigenvalue,
            "problematic_ells": self.problematic_ells,
            "positive_semidefinite": self.positive_semidefinite,
            "ok": self.ok,
        }


@dataclass(frozen=True)
class MockValidationReport:
    """Summary statistics comparing generated mocks to a calibration object."""

    n_mocks: int
    n_bins: int
    n_pix: int
    finite: bool
    mean_by_bin: np.ndarray
    target_mean_by_bin: np.ndarray | None
    std_by_bin: np.ndarray
    target_std_by_bin: np.ndarray | None
    histogram_l1_by_bin: np.ndarray | None
    spectra_ratio_mean: np.ndarray | None
    spectra_ratio_std: np.ndarray | None

    @property
    def spectra_available(self) -> bool:
        """Whether spectra ratios were computed."""

        return self.spectra_ratio_mean is not None

    @property
    def ok(self) -> bool:
        """Basic finite-value pass/fail flag for generated mocks."""

        return self.finite

    def as_dict(self) -> dict[str, object]:
        """Return report values in a dictionary for notebooks or logs."""

        return {
            "n_mocks": self.n_mocks,
            "n_bins": self.n_bins,
            "n_pix": self.n_pix,
            "finite": self.finite,
            "mean_by_bin": self.mean_by_bin,
            "target_mean_by_bin": self.target_mean_by_bin,
            "std_by_bin": self.std_by_bin,
            "target_std_by_bin": self.target_std_by_bin,
            "histogram_l1_by_bin": self.histogram_l1_by_bin,
            "spectra_ratio_mean": self.spectra_ratio_mean,
            "spectra_ratio_std": self.spectra_ratio_std,
            "spectra_available": self.spectra_available,
            "ok": self.ok,
        }


def validate_kappa_maps(kappa_maps, *, nside: int | None = None) -> np.ndarray:
    """Return kappa maps as ``(n_bins, n_pix)`` float arrays."""

    maps = np.asarray(kappa_maps, dtype=float)
    if maps.ndim == 1:
        maps = maps[np.newaxis, :]
    if maps.ndim != 2:
        raise ValueError("kappa_maps must have shape (n_bins, n_pix) or (n_pix,).")
    if maps.shape[1] == 0:
        raise ValueError("kappa_maps must contain at least one pixel.")
    if not np.all(np.isfinite(maps)):
        raise ValueError("kappa_maps must contain only finite values.")
    if nside is not None:
        expected_npix = 12 * int(nside) * int(nside)
        if maps.shape[1] != expected_npix:
            raise ValueError(
                f"kappa_maps has {maps.shape[1]} pixels, but nside={nside} "
                f"requires {expected_npix} HEALPix pixels."
            )
    return maps


def spectrum_diagnostics(
    cl,
    *,
    n_bins: int | None = None,
    name: str = "cl",
    symmetry_atol: float = 1e-12,
    psd_atol: float = 1e-12,
) -> SpectrumDiagnostics:
    """Return finite/symmetry/semidefinite diagnostics for spectra."""

    arr = np.asarray(cl, dtype=float)
    shape = tuple(arr.shape)
    finite = bool(np.all(np.isfinite(arr)))
    symmetric = False
    min_eigenvalue = np.nan
    problematic_ells: tuple[int, ...] = ()

    if arr.ndim == 3 and arr.shape[0] == arr.shape[1]:
        if n_bins is None or arr.shape[0] == n_bins:
            symmetric = bool(np.allclose(arr, np.swapaxes(arr, 0, 1), atol=symmetry_atol))
            if finite:
                min_eigs = np.array(
                    [
                        np.linalg.eigvalsh(arr[:, :, ell]).min()
                        for ell in range(arr.shape[2])
                    ]
                )
                min_eigenvalue = float(min_eigs.min()) if min_eigs.size else np.nan
                problematic_ells = tuple(int(ell) for ell in np.flatnonzero(min_eigs < -psd_atol))

    return SpectrumDiagnostics(
        name=name,
        shape=shape,
        finite=finite,
        symmetric=symmetric,
        min_eigenvalue=min_eigenvalue,
        problematic_ells=problematic_ells,
    )


def validate_cls(
    cl,
    *,
    n_bins: int,
    name: str = "cl_ng",
    check_symmetric: bool = True,
    check_psd: bool = True,
    symmetry_atol: float = 1e-12,
    psd_atol: float = 1e-12,
) -> np.ndarray:
    """Return spectra as ``(n_bins, n_bins, lmax + 1)`` float arrays."""

    arr = np.asarray(cl, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (n_bins, n_bins, lmax + 1).")
    if arr.shape[0] != n_bins or arr.shape[1] != n_bins:
        raise ValueError(
            f"{name} has bin shape {arr.shape[:2]}, expected ({n_bins}, {n_bins})."
        )
    if arr.shape[2] < 2:
        raise ValueError(f"{name} must contain at least ell=0 and ell=1.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    if check_symmetric and not np.allclose(arr, np.swapaxes(arr, 0, 1), atol=symmetry_atol):
        raise ValueError(f"{name} must be symmetric across tomographic bin axes.")
    if check_psd:
        diagnostics = spectrum_diagnostics(
            arr,
            n_bins=n_bins,
            name=name,
            symmetry_atol=symmetry_atol,
            psd_atol=psd_atol,
        )
        if diagnostics.problematic_ells:
            first = diagnostics.problematic_ells[:5]
            raise ValueError(
                f"{name} must be positive semidefinite at each ell; "
                f"min eigenvalue={diagnostics.min_eigenvalue:.6e}, "
                f"problematic ell values start with {first}."
            )
    return arr


def validate_pixwin(pixwin, *, lmax: int | None = None) -> np.ndarray:
    """Return a one-dimensional pixel window array."""

    arr = np.asarray(pixwin, dtype=float)
    if arr.ndim != 1:
        raise ValueError("pixwin must have shape (lmax + 1,).")
    if arr.size == 0:
        raise ValueError("pixwin must contain at least one value.")
    if lmax is not None and arr.size < lmax + 1:
        raise ValueError(f"pixwin has length {arr.size}, but ell={lmax} is required.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("pixwin must contain only finite values.")
    return arr
