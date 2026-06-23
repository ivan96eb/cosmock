"""Point transformations from latent Gaussian fields to kappa fields."""

from __future__ import annotations

import numpy as np

from .quadrature import get_gh_nodes_weights


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

    if n == "4":
        a1, a2, t, x0 = params
        arg1 = np.exp(a1 * x - 0.5 * a1**2)
        arg2 = (1 + np.exp((x - x0) * t)) ** ((a2 - a1) / t)
        return compute_normalization() * arg1 * arg2 - 1

    a1, a2, b, t, x0 = params
    arg1 = np.exp(a1 * x - 0.5 * a1**2) + b * x
    arg2 = (1 + np.exp((x - x0) * t)) ** ((a2 - a1) / t)
    return compute_normalization() * arg1 * arg2 - 1


def Gn(x, n, params, N_nodes: int = 20):
    """Compatibility alias for the legacy GPTG transform function."""

    return gptg_transform(x, n, params, n_nodes=N_nodes)

