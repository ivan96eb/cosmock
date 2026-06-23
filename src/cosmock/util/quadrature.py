"""Quadrature utilities used by GPTG transformations."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.hermite import hermgauss


def get_gh_nodes_weights(n_nodes: int):
    """Return nodes and weights for expectations over a standard normal."""

    t, w = hermgauss(n_nodes)
    y_nodes = np.sqrt(2.0) * t
    y_weights = w / np.sqrt(np.pi)
    return y_nodes, y_weights
