"""Fitting routines for GPTG transform parameters."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .transforms import Gn


def variance_from_Cl(Cl, ell_min=0):
    """Compute field variance predicted by an angular power spectrum."""

    Cl = np.asarray(Cl)
    ell = np.arange(ell_min, ell_min + len(Cl))
    return np.sum((2 * ell + 1) * Cl) / (4 * np.pi)


def fit_gn(x_data, y_data, N, initial_params=None):
    """Fit an unconstrained GPTG transform to binned ``(x, y)`` data."""

    N = str(N)
    if initial_params is None:
        initial_params = np.ones(int(N))

    def cost_function(params):
        try:
            y_pred = Gn(x_data, N, params)
            return np.sum((y_pred - y_data) ** 2)
        except Exception:
            return np.inf

    result = minimize(fun=cost_function, x0=initial_params, method="BFGS")
    return result.x


def fit_gn_with_constraint(x_data, y_data, N, cls, initial_params=None):
    """Fit a GPTG transform with variance constrained by a target spectrum."""

    N = str(N)
    var = variance_from_Cl(cls)

    def calc_constrained_params(unconstrained_params, var, N):
        if N == "2":
            beta = unconstrained_params[0]
            alpha = np.sqrt(np.log(1 + var / beta**2))
            return np.array([alpha, beta])
        if N == "3":
            a, b = unconstrained_params
            c = np.sqrt((np.exp(a**2) - 1 + 2 * a * b + b**2) / var) - 1
            return np.array([a, b, c])
        raise ValueError(f"Cannot do a constrained fit for G{N}. Use fit_gn instead.")

    def cost_function(unconstrained_params):
        params = calc_constrained_params(unconstrained_params, var, N)
        try:
            y_pred = Gn(x_data, N, params)
            return np.sum((y_pred - y_data) ** 2)
        except Exception:
            return np.inf

    if initial_params is None:
        if N == "2":
            beta_grid = np.geomspace(1e-6, 1, 50)
            cost_grid = np.array([cost_function([b]) for b in beta_grid])
            beta_init = beta_grid[np.argmin(cost_grid)]
            initial_params = np.array([beta_init, np.nan])
        elif N == "3":
            a_grid = np.linspace(0.01, 1.7, 25)
            b_grid = np.linspace(0.01, 2.7, 25)
            cost_grid = np.array(
                [[cost_function([aa, bb]) for aa in a_grid] for bb in b_grid]
            )
            j_min, i_min = np.unravel_index(np.argmin(cost_grid), cost_grid.shape)
            initial_params = np.array([a_grid[i_min], b_grid[j_min], np.nan])
        else:
            initial_params = np.ones(int(N))

    initial_unconstrained_params = initial_params[: int(N) - 1]
    result = minimize(fun=cost_function, x0=initial_unconstrained_params, method="BFGS")
    return calc_constrained_params(result.x, var, N)


def fit_transform(calibration, *, order=3, constrained: bool = True, initial_params=None):
    """Fit GPTG transform parameters for every tomographic bin in a calibration."""

    order = str(order)
    transform_params = np.zeros((calibration.n_bins, int(order)), dtype=float)
    can_constrain = constrained and calibration.cl_ng is not None and order in {"2", "3"}

    for i in range(calibration.n_bins):
        x_data = calibration.binned_x[i]
        y_data = calibration.binned_y[i]
        init_i = None
        if initial_params is not None:
            init_i = np.asarray(initial_params)[i]

        if can_constrain:
            transform_params[i] = fit_gn_with_constraint(
                x_data, y_data, order, calibration.cl_ng[i, i], initial_params=init_i
            )
        else:
            transform_params[i] = fit_gn(x_data, y_data, order, initial_params=init_i)

    return transform_params

