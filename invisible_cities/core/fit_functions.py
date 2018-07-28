"""
A set of functions for data fitting.

GML November 2016
"""

import numpy  as np
import pandas as pd
import scipy.optimize
import scipy.stats

from .                    import core_functions as coref
from .  stat_functions    import poisson_sigma
from .. evm.ic_containers import FitFunction


def get_errors(cov):
    """
    Find errors from covariance matrix

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix of the fit parameters.

    Returns
    -------
    err : 1-dim np.ndarray
        Errors asociated to the fit parameters.
    """
    return np.sqrt(np.diag(cov))


def get_chi2_and_pvalue(ydata, yfit, ndf, sigma=None):
    """
    Gets reduced chi2 and p-value

    Parameters
    ----------
    ydata : np.ndarray
        Data points.
    yfit : np.ndarray
        Fit values corresponding to ydata array.
    sigma : np.ndarray
        Data errors. If sigma is not given, it takes the poisson case:
            sigma = sqrt(ydata)
    ndf : int
        Number of degrees of freedom
        (number of data points - number of parameters).

    Returns
    -------
    chi2 : float
        Reduced chi2 computed as:
            chi2 = [sum(ydata - yfit)**2 / sigma**2] / ndf
    pvalue : float
        Fit p-value.
    """

    if sigma is None:
        sigma = poisson_sigma(ydata)

    chi2   = np.sum(((ydata - yfit) / sigma)**2)
    pvalue = scipy.stats.chi2.sf(chi2, ndf)

    return chi2 / ndf, pvalue


# ###########################################################
# Functions
def gauss(x, amp, mu, sigma):
    if sigma <= 0.:
        return np.inf
    return amp/(2*np.pi)**.5/sigma * np.exp(-0.5*(x-mu)**2./sigma**2.)


def polynom(x, *coeffs):
    return np.polynomial.polynomial.polyval(x, coeffs)


def expo(x, const, mean):
    return const * np.exp(x/mean)


def power(x, const, pow_):
    return const * np.power(x, pow_)


# ###########################################################
# Tools
def fit(func, x, y, seed=(), fit_range=None, **kwargs):
    """
    Fit x, y data to a generic relation of already defined
    python functions.

    Parameters
    ----------
    func : function
        A callable object with signature (x, par0, par1, ...) where x is
        the value (or array of values) at which the function is evaluated
        and par represent the coefficients of the function.
    x, y : iterables
        Data sets to be fitted.
    seed : sequence
        Initial estimation of the fit parameters. Either all or none of them
        must be given.
    fit_range : tuple
        Range of x in which the fit is performed.
    Notes
    -----
    - Functions must be vectorized.

    Returns
    -------
    fitted_fun : extended function (contains values and errors)
        Fitted function.

    Examples
    --------
    >>> import numpy as np
    >>> import invisible_cities.core.fit_functions as fit
    >>> x = np.linspace(-5, 5, 100)
    >>> y = np.exp(-(x-1.)**2)
    >>> f = fit.fit(fit.gauss, x, y, (1., 2., 3))
    >>> print(f.values)
    [ 1.77245385  1.          0.70710678]
    """
    if fit_range is not None:
        sel  = coref.in_range(x, *fit_range)
        x, y = x[sel], y[sel]
        if "sigma" in kwargs:
            kwargs["sigma"] = kwargs["sigma"][sel]

    sigma_r = kwargs.get("sigma", np.ones_like(y))
    if np.any(sigma_r <= 0):
        raise ValueError("Zero or negative value found in argument sigma. "
                         "Errors must be greater than 0.")

    kwargs['absolute_sigma'] = "sigma" in kwargs

    vals, cov = scipy.optimize.curve_fit(func, x, y, seed, **kwargs)

    fitf       = lambda x: func(x, *vals)
    fitx       = fitf(x)
    errors     = get_errors(cov)
    ndof       = len(y) - len(vals)
    chi2, pval = get_chi2_and_pvalue(y, fitx, ndof, sigma_r)


    return FitFunction(fitf, vals, errors, chi2, pval, cov)


def profileX(xdata, ydata, nbins=100,
             xrange=None, yrange=None,
             std=False, drop_nan=True):
    """
    Compute the x-axis binned average of a dataset.

    Parameters
    ----------
    xdata, ydata : 1-dim np.ndarray
        x and y coordinates from a dataset.
    nbins : int, optional
        Number of divisions in the x axis. Defaults to 100.
    xrange : tuple of ints/floats or None, optional
        Range over the x axis. Defaults to dataset extremes.
    yrange : tuple of ints/floats or None, optional
        Range over the y axis. Defaults to dataset extremes.
    drop_nan : bool, optional
        Exclude empty bins. Defaults to True.

    Returns
    -------
    x_out : 1-dim np.ndarray.
        Bin centers.
    y_out : 1-dim np.ndarray
        Data average for each bin.
    y_err : 1-dim np.ndarray
        Average error for each bin.
    """
    if xrange is None: xrange = np.min(xdata), np.max(xdata)

    selection = coref.in_range(xdata, *xrange)
    xdata     = xdata[selection]
    ydata     = ydata[selection]

    if yrange is not None:
        selection = coref.in_range(ydata, *yrange)
        xdata     = xdata[selection]
        ydata     = ydata[selection]

    bin_edges   = np.linspace(*xrange, nbins + 1)
    bin_centers = coref.shift_to_bin_centers(bin_edges)
    bin_numbers = np.digitize(xdata, bin_edges, right=False)
    df          = pd.DataFrame(dict(bin=bin_numbers, y=ydata))
    gb          = df.groupby("bin").y

    mean      = gb.mean().values
    deviation = gb.std () if std else gb.std() / gb.size()**0.5
    indices   = deviation.index.values - 1

    if drop_nan:
        return bin_centers[indices], mean, deviation.values

    mean_               = np.full_like(bin_centers, np.nan)
    deviation_          = np.full_like(bin_centers, np.nan)
    mean_     [indices] = mean
    deviation_[indices] = deviation

    return bin_centers, mean_, deviation_


def profileY(xdata, ydata, nbins = 100,
             yrange=None, xrange=None,
             std=False, drop_nan=True):
    """
    Compute the y-axis binned average of a dataset.

    Parameters
    ----------
    xdata, ydata : 1-dim np.ndarray
        x and y coordinates from a dataset.
    nbins : int
        Number of divisions in the y axis.
    yrange : tuple of ints/floats or None, optional
        Range over the y axis. Defaults to dataset extremes.
    xrange : tuple of ints/floats or None, optional
        Range over the x axis. Defaults to dataset extremes.
    drop_nan : bool, optional
        Exclude empty bins. Defaults to True.

    Returns
    -------
    x_out : 1-dim np.ndarray.
        Bin centers.
    y_out : 1-dim np.ndarray
        Data average for each bin.
    y_err : 1-dim np.ndarray
        Average error for each bin.
    """
    return profileX(ydata, xdata, nbins, yrange, xrange, std, drop_nan)


def profileXY(xdata, ydata, zdata, nbinsx, nbinsy,
              xrange=None, yrange=None, zrange=None,
              std=False, drop_nan=True):
    """
    Compute the xy-axis binned average of a dataset.

    Parameters
    ----------
    xdata, ydata, zdata : 1-dim np.ndarray
        x, y, z coordinates from a dataset.
    nbinsx, nbinsy : int
        Number of divisions in each axis.
    xrange : tuple of ints/floats or None, optional
        Range over the x axis. Defaults to dataset extremes.
    yrange : tuple of ints/floats or None, optional
        Range over the y axis. Defaults to dataset extremes.
    zrange : tuple of ints/floats or None, optional
        Range over the z axis. Defaults to dataset extremes.
    drop_nan : bool, optional
        Exclude empty bins. Defaults to True.

    Returns
    -------
    x_out : 1-dim np.ndarray.
        Bin centers in the x axis.
    y_out : 1-dim np.ndarray.
        Bin centers in the y axis.
    z_out : 1-dim np.ndarray
        Data average for each bin.
    z_err : 1-dim np.ndarray
        Average error for each bin.
    """
    if xrange is None: xrange = np.min(xdata), np.max(xdata)
    if yrange is None: yrange = np.min(ydata), np.max(ydata)

    selection  = coref.in_range(xdata, *xrange)
    selection &= coref.in_range(ydata, *yrange)
    xdata      = xdata[selection]
    ydata      = ydata[selection]
    zdata      = zdata[selection]

    if zrange is not None:
        selection = coref.in_range(zdata, *zrange)
        xdata     = xdata[selection]
        ydata     = ydata[selection]
        zdata     = zdata[selection]

    bin_edges_x   = np.linspace(*xrange, nbinsx + 1)
    bin_edges_y   = np.linspace(*yrange, nbinsy + 1)
    bin_centers_x = coref.shift_to_bin_centers(bin_edges_x)
    bin_centers_y = coref.shift_to_bin_centers(bin_edges_y)
    bin_numbers_x = np.digitize(xdata, bin_edges_x, right=False)
    bin_numbers_y = np.digitize(ydata, bin_edges_y, right=False)
    df            = pd.DataFrame(dict(binx=bin_numbers_x, biny=bin_numbers_y, z=zdata))
    gb            = df.groupby(["binx", "biny"]).z

    shape     = nbinsx, nbinsy
    mean      = np.zeros(shape)
    deviation = np.zeros(shape)

    mean_       = gb.mean().values
    deviation_  = gb.std () if std else gb.std() / gb.size()**0.5
    (indices_x,
     indices_y) = map(np.array, zip(*deviation_.index.values))

    notnan = ~np.isnan(deviation_.values)
    mean     [indices_x - 1, indices_y - 1] = mean_
    deviation[indices_x - 1, indices_y - 1] = np.where(notnan, deviation_.values, 0)

    return bin_centers_x, bin_centers_y, mean, deviation
