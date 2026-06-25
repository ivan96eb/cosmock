"""Analytical point transformations used by cosmock."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_order(order: int) -> int:
    """Validate and return a supported transformation order."""

    if not isinstance(order, int):
        raise TypeError("order must be an integer.")
    if order not in {2, 3}:
        raise ValueError("order must be either 2 or 3.")
    return order


def evaluate_transform(
    x: ArrayLike,
    order: int,
    transform_params: ArrayLike,
) -> NDArray[np.float64]:
    """Evaluate a fitted point transformation.

    Parameters
    ----------
    x : array_like
        Standard-normal values from the latent Gaussian field.
    order : {2, 3}
        Transformation order.
    transform_params : array_like
        For ``order=2``, ``(alpha, beta)``. For ``order=3``,
        ``(a, b, c)``.

    Returns
    -------
    ndarray
        Transformed scalar-field values with the same broadcast shape as ``x``.

    Raises
    ------
    TypeError
        If ``order`` is not an integer.
    ValueError
        If ``order`` is unsupported or the parameter length is invalid.
    """

    order = validate_order(order)
    params = np.asarray(transform_params, dtype=float)
    if params.shape != (order,):
        raise ValueError(
            f"transform_params must have shape ({order},) for order={order}."
        )

    x_array = np.asarray(x, dtype=float)
    if order == 2:
        alpha, beta = params
        return beta * np.exp(alpha * x_array - 0.5 * alpha**2) - beta

    a, b, c = params
    transformed = np.exp(a * x_array - 0.5 * a**2) + b * x_array + c
    return transformed / (1.0 + c) - 1.0
