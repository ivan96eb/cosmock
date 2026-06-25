"""Analytical point-transformation models used by cosmock."""

import numpy as np

def Gn(x, N, lbda):
    """Evaluate a fitted ``G_N`` transformation.

    Parameters
    ----------
    x : array_like
        Standard-normal input values from the latent Gaussian field.
    N : int
        Transformation order. The current implementation supports ``2`` and
        ``3``.
    lbda : array_like
        Transformation parameters. For ``N=2`` this is ``(alpha, beta)``.
        For ``N=3`` this is ``(a, b, c)``.

    Returns
    -------
    numpy.ndarray
        Transformed values with the same broadcast shape as ``x``.

    Raises
    ------
    ValueError
        If ``N`` is not ``2`` or ``3``.
    """
    # Evaluate transformation
    if N == 2:
        alpha, beta = lbda
        return beta * np.exp(alpha * x - 0.5 * alpha**2) - beta
    
    elif N == 3:
        a, b, c = lbda
        arg = np.exp(a * x - 0.5 * a**2) + b*x + c
        norm = 1/(1+c)
        return norm * arg - 1
    
    else:
        raise ValueError(f"Unknown model type: {N}")
