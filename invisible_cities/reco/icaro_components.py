import numpy        as np

from   typing              import Optional


def lin_seed(x : np.array,
             y : np.array):

    '''
    Estimate the seed for a linear fit.

    Parameters
    ----------
    x : np.array
        Independent variable.
    y : np.array
        Dependent variable.

    Returns
    -------
    seed : tuple
        Seed parameters (intercept, slope) for the linear fit.
    '''


    x0, x1 = x.min(), x.max()
    y0, y1 = y.min(), y.max()

    b = (y1 - y0) / (x1 - x0)
    a = y0 - b * x0

    seed = a, b

    return seed



def expo_seed(x   : np.array,
              y   : np.array,
              eps : Optional[float] = 1e-12):

    '''
    Estimate the seed for an exponential fit.

    Parameters
    ----------
    x : np.array
        Independent variable.
    y : np.array
        Dependent variable.
    eps : float, optional
        Small value added to prevent division by zero, default is 1e-12.

    Returns
    -------
    seed : tuple
        Seed parameters (constant, mean) for the exponential fit.
    '''

    x, y  = zip(*sorted(zip(x, y)))

    const = y[0]
    slope = (x[-1] - x[0]) / np.log(y[-1] / (y[0] + eps))
    seed  = const, slope

    return seed

