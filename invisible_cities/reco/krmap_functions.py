import numpy  as np
import pandas as pd

from .. types.symbols       import KrFitFunction
from .. evm.ic_containers   import FitFunction
from .. core.fit_functions  import polynom, expo


def lin_seed(x : np.array, y : np.array):
    '''
    Estimate the seed for a linear fit of the form y = a + bx.

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

    if x1 == x0: # If same x value, set slope to 0 and use the mean value of y as interceipt
        b = 0
        a = y.mean()
    else:
        b = (y1 - y0) / (x1 - x0)
        a = y0 - b * x0

    return a, b


def expo_seed(x : np.array, y : np.array):
    '''
    Estimate the seed for an exponential fit of the form y = y0*exp(-x/lt).

    Parameters
    ----------
    x : np.array
        Independent variable.
    y : np.array
        Dependent variable.

    Returns
    -------
    seed : tuple
        Seed parameters (y0, lt) for the exponential fit.
    '''
    x, y = zip(*sorted(zip(x, y)))
    y0   = y[0]

    if y0 <= 0 or y[-1] <= 0:
        raise ValueError("y data must be > 0")

    lt = -x[-1] / np.log(y[-1] / y0)

    return y0, lt


def select_fit_variables(fittype : KrFitFunction, dst : pd.DataFrame):
    '''
    Select the data for fitting based on the specified fit type.

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
    if   fittype is KrFitFunction.linear : return dst.DT,         dst.S2e
    elif fittype is KrFitFunction.expo   : return dst.DT,         dst.S2e
    elif fittype is KrFitFunction.log_lin: return dst.DT, -np.log(dst.S2e)


def get_function_and_seed_lt(fittype : KrFitFunction):
    '''
    Retrieve the fitting function and seed function based on the
    specified fittype.

    Parameters
    ----------
    fittype : KrFitFunction
        The type of fit function to retrieve (e.g., linear, exponential, log-linear).

    Returns
    -------
    fit_function  : function
        The fitting function corresponding to the specified fit type.
    seed_function : function
        The seed function corresponding to the specified fit type.
    '''
    linear_function  = lambda x, y0, slope: polynom(x, y0, slope)
    expo_function    = lambda x, e0, lt:    expo   (x, e0, -lt)

    if   fittype is KrFitFunction.linear:  return linear_function,  lin_seed
    elif fittype is KrFitFunction.log_lin: return linear_function,  lin_seed
    elif fittype is KrFitFunction.expo:    return   expo_function, expo_seed


def transform_parameters(fit_output : FitFunction):
    '''
    Transform the parameters obtained from the fitting output into EO and LT.
    When using log_lin fit, we need to convert the intermediate variables into
    the actual physical magnitudes involved in the process.

    Parameters
    ----------
    fit_output : FitFunction
        Output from IC's fit containing the parameter values, errors, and
        covariance matrix.

    Returns
    -------
    par : list
        Transformed parameter values.
    err : list
        Transformed parameter errors.
    cov : float
        Transformed covariance value.
    '''
    par = fit_output.values
    err = fit_output.errors
    cov = fit_output.cov[0, 1]

    a, b     = par
    u_a, u_b = err

    E0   = np.exp(-a)
    s_E0 = np.abs(E0 * u_a)
    lt   = 1 / b
    s_lt = np.abs(lt**2 * u_b)
    cov  = E0 * lt**2 * cov # Not sure about this

    par  = [  E0,   lt]
    err  = [s_E0, s_lt]

    return par, err, cov

