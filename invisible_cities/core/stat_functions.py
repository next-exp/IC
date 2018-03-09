"""
Contains statistics-related functions.
"""

import numpy as np


def poisson_factor(k, mean):
    """
    Probability mass function for a poisson statistics.
    Faster than scipy.stats.poisson.pmf.
    """
    return mean ** k * np.exp(-mean) / np.math.factorial(k)


def poisson_sigma(x, default=3):
    """
    Get the uncertainty of x (assuming it is poisson-distributed).
    Set *default* when x is 0 to avoid null uncertainties.
    """
    u = x**0.5
    u[x==0] = default
    return u
