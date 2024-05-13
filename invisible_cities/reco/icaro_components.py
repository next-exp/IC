import numpy        as np
import pandas       as pd

from   typing              import Optional
from ..types.symbols       import KrFitFunction

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


def prepare_data(fittype : KrFitFunction,
                 dst     : pd.DataFrame):

    '''
    Prepare the data for fitting based on the specified fit type.

    NOTES: Since x axis (DT) is never altered, maybe we can just
    return the y values. However, when we implement the binned fit,
    the profile could be done here (?) so it would make sense to
    always provide both x and y. We could rename parameters and have
    fittype (binned / unbinned) and fitfunction (lin, expo, log-lin...)

    Parameters
    ----------
    fittype : KrFitFunction
        The type of fit function to prepare data for (e.g., linear, exponential, log-linear).
    dst : pd.DataFrame
        The DataFrame containing the data to be prepared for fitting.

    Returns
    -------
    x_data : pd.Series
        The independent variable data prepared for fitting.
    y_data : pd.Series
        The dependent variable data prepared for fitting.
    '''

    if fittype is KrFitFunction.linear:
        return dst.DT, dst.S2e

    elif fittype is KrFitFunction.expo:
        return dst.DT, dst.S2e

    elif fittype is KrFitFunction.log_lin:
        return dst.DT, -np.log(dst.S2e)
