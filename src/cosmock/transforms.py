"""Point transformations from latent Gaussian fields to kappa fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .util.quadrature import get_gh_nodes_weights


def _order_as_str(order) -> str:
    order_str = str(order)
    if order_str not in {"2", "3", "4", "5"}:
        raise ValueError(f"Unknown GPTG order: {order}")
    return order_str


def gptg_transform(x, order, params, n_nodes: int = 20):
    """Evaluate a GPTG ``G_N`` transformation at standard-normal values ``x``."""

    n = _order_as_str(order)
    x = np.asarray(x)
    params = np.asarray(params, dtype=float)

    def compute_normalization():
        if n in {"2", "3"}:
            return None
        if n == "4":
            a1, a2, t, x0 = params
            quad_points, quad_weights = get_gh_nodes_weights(n_nodes)
            arg1 = np.exp(a1 * quad_points - 0.5 * a1**2)
            arg2 = (1 + np.exp((quad_points - x0) * t)) ** ((a2 - a1) / t)
            return 1 / np.sum(quad_weights * arg1 * arg2)
        a1, a2, b, t, x0 = params
        quad_points, quad_weights = get_gh_nodes_weights(n_nodes)
        arg1 = np.exp(a1 * quad_points - 0.5 * a1**2) + b * quad_points
        arg2 = (1 + np.exp((quad_points - x0) * t)) ** ((a2 - a1) / t)
        return 1 / np.sum(quad_weights * arg1 * arg2)

    if n == "2":
        alpha, beta = params
        return beta * np.exp(alpha * x - 0.5 * alpha**2) - beta

    if n == "3":
        a, b, c = params
        arg = np.exp(a * x - 0.5 * a**2) + b * x + c
        norm = 1 / (1 + c)
        return norm * arg - 1

    # if n == "4":
    #     a1, a2, t, x0 = params
    #     arg1 = np.exp(a1 * x - 0.5 * a1**2)
    #     arg2 = (1 + np.exp((x - x0) * t)) ** ((a2 - a1) / t)
    #     return compute_normalization() * arg1 * arg2 - 1
    # if n == "5":
        # a1, a2, b, t, x0 = params
        # arg1 = np.exp(a1 * x - 0.5 * a1**2) + b * x
        # arg2 = (1 + np.exp((x - x0) * t)) ** ((a2 - a1) / t)
        # return compute_normalization() * arg1 * arg2 - 1
    else:
        raise ValueError(f"Unknown GPTG order: {n}")


@dataclass(frozen=True)
class GPTGTransform:
    """A fitted GPTG point transform for one tomographic bin."""

    order: str
    params: np.ndarray

    def __post_init__(self):
        order = _order_as_str(self.order)
        params = np.asarray(self.params, dtype=float)
        if params.ndim != 1:
            raise ValueError("params must be one-dimensional for one GPTG transform.")
        if params.shape[0] != int(order):
            raise ValueError(f"G{order} expects {order} parameters, got {params.shape[0]}.")
        if not np.all(np.isfinite(params)):
            raise ValueError("params must contain only finite values.")
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "params", params)

    def evaluate(self, x, *, n_nodes: int = 20):
        """Evaluate this fitted transform at standard-normal values ``x``."""

        return gptg_transform(x, self.order, self.params, n_nodes=n_nodes)


@dataclass(frozen=True)
class GPTGTransformSet:
    """Fitted GPTG point transforms for all tomographic bins."""

    order: str
    params: np.ndarray
    calibration: object | None = None
    constrained: bool = True
    fit_settings: dict[str, object] | None = None

    def __post_init__(self):
        order = _order_as_str(self.order)
        params = np.asarray(self.params, dtype=float)
        if params.ndim != 2:
            raise ValueError("params must have shape (n_bins, n_params).")
        if params.shape[1] != int(order):
            raise ValueError(f"G{order} expects {order} parameters per bin.")
        if not np.all(np.isfinite(params)):
            raise ValueError("params must contain only finite values.")
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "params", params)

    @property
    def n_bins(self) -> int:
        """Number of fitted tomographic bins."""

        return int(self.params.shape[0])

    @property
    def transform_params(self) -> np.ndarray:
        """Array view of fitted parameters with shape ``(n_bins, n_params)``."""

        return self.params

    def __len__(self) -> int:
        return self.n_bins

    def __getitem__(self, bin_index: int) -> GPTGTransform:
        return GPTGTransform(self.order, self.params[bin_index])

    def evaluate(self, x, *, bin_index: int, n_nodes: int = 20):
        """Evaluate one bin's fitted transform at standard-normal values ``x``."""

        return self[bin_index].evaluate(x, n_nodes=n_nodes)

    def to_latent_spectra(self, cl_ng=None, *, n_jobs: int = 4, **spectra_kwargs):
        """Convert target kappa spectra to latent Gaussian spectra."""

        if cl_ng is None:
            if self.calibration is None or getattr(self.calibration, "cl_ng", None) is None:
                raise ValueError("cl_ng is required when the transform set has no calibration.")
            cl_ng = self.calibration.cl_ng

        from .spectra import target_cls_to_latent_cls

        return target_cls_to_latent_cls(
            cl_ng,
            self.params,
            order=self.order,
            n_jobs=n_jobs,
            **spectra_kwargs,
        )


def Gn(x, n, params, N_nodes: int = 20):
    """Compatibility alias for the legacy GPTG transform function."""

    return gptg_transform(x, n, params, n_nodes=N_nodes)
