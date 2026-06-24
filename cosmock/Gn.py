import numpy as np
from .structs import NonlinParameters


def Gn(x, params):
    """
    Evaluate Gn transformation at given x values
    
    Parameters
    ----------
    x : array of x values (standard normal inputs)
    n : str, which G function to use ('2', '3', '4', '5')
    params : array of parameters for the transformation
    N_nodes : number of Gauss-Hermite quad points to compute
             integral for normalization.
    
    Returns
    -------
    y : transformed values
    """
    # Evaluate transformation
    if params.N == 2:
        alpha, beta = params.lbda
        return beta * np.exp(alpha * x - 0.5 * alpha**2) - beta
    
    elif params.N == 3:
        a, b, c = params.lbda
        arg = np.exp(a * x - 0.5 * a**2) + b*x + c
        norm = 1/(1+c)
        return norm * arg - 1
    
    else:
        raise ValueError(f"Unknown model type: {params.N}")