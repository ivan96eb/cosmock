from typing import NamedTuple
import numpy as np

class NonlinParameters(NamedTuple):
    lbda: np.ndarray
    N: int
    Cl_Gauss: np.ndarray