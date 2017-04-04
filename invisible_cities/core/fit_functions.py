"""
A set of functions for data fitting.

GML November 2016
"""

from __future__ import absolute_import, division

import numpy as np
import scipy.optimize

from invisible_cities.reco.params import FitFunction

def in_range(data, minval=-np.inf, maxval=np.inf):
    """
    Find values in range [minval, maxval).

    Parameters
    ---------
    data : np.ndarray
        Data set of arbitrary dimension.
    minval : int or float, optional
        Range minimum. Defaults to -inf.
    maxval : int or float, optional
        Range maximum. Defaults to +inf.

    Returns
    -------
    selection : np.ndarray
        Boolean array with the same dimension as the input. Contains True
        for those values of data in the input range and False for the others.
    """
    return (minval <= data) & (data < maxval)


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
        sel  = in_range(x, *fit_range)
        x, y = x[sel], y[sel]

    vals, cov = scipy.optimize.curve_fit(func,
                                         x, y,
                                         seed,
                                         **kwargs)

    fitf = lambda x: func(x, *vals)
    fitx = fitf(x)
    chi2 = np.sum(np.ma.masked_invalid((fitx - y)**2/y))

    return FitFunction(fitf,
                       vals,
                       get_errors(cov),
                       chi2 / (len(x) - len(vals)))


def profileX(xdata, ydata, nbins=100,
             xrange=None, yrange=None, drop_nan=True):
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
    xmin, xmax = xrange if xrange else (np.min(xdata), np.max(xdata))
    ymin, ymax = yrange if yrange else (np.min(ydata), np.max(ydata))

    x_out = np.linspace(xmin, xmax, nbins+1)
    y_out = np.empty(nbins)
    y_err = np.empty(nbins)
    dx    = x_out[1] - x_out[0]

    selection = in_range(xdata, xmin, xmax) & in_range(ydata, ymin, ymax)
    x, y = xdata[selection], ydata[selection]
    for i in range(nbins):
        bin_data = y[in_range(x,
                              minval = x_out[i],
                              maxval = x_out[i+1])]
        y_out[i] = np.mean(bin_data)
        y_err[i] = np.std(bin_data) / bin_data.size ** 0.5
    x_out += dx / 2.
    x_out  = x_out[:-1]
    if drop_nan:
        selection = ~(np.isnan(y_out) | np.isnan(y_err))
        x_out = x_out[selection]
        y_out = y_out[selection]
        y_err = y_err[selection]
    return x_out, y_out, y_err


def profileY(xdata, ydata, nbins = 100,
             yrange=None, xrange=None, drop_nan=True):
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
    return profileX(ydata, xdata, nbins, yrange, xrange, drop_nan)


def profileXY(xdata, ydata, zdata, nbinsx, nbinsy,
              xrange=None, yrange=None, zrange=None, drop_nan=True):
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
    xmin, xmax = xrange if xrange else (np.min(xdata), np.max(xdata))
    ymin, ymax = yrange if yrange else (np.min(ydata), np.max(ydata))
    zmin, zmax = zrange if zrange else (np.min(zdata), np.max(zdata))

    x_out = np.linspace(xmin, xmax, nbinsx+1)
    y_out = np.linspace(ymin, ymax, nbinsy+1)
    z_out = np.empty((nbinsx, nbinsy))
    z_err = np.empty((nbinsx, nbinsy))
    dx = x_out[1] - x_out[0]
    dy = y_out[1] - y_out[0]

    selection = (in_range(xdata, xmin, xmax) &
                 in_range(ydata, ymin, ymax) &
                 in_range(zdata, zmin, zmax))
    xdata, ydata, zdata = xdata[selection], ydata[selection], zdata[selection]
    for i in range(nbinsx):
        for j in range(nbinsy):
            selection = (in_range(xdata, x_out[i], x_out[i+1]) &
                         in_range(ydata, y_out[j], y_out[j+1]))
            bin_data = zdata[selection]
            z_out[i,j] = np.nanmean(bin_data) if bin_data.size else 0.
            z_err[i,j] = np.nanstd(bin_data) / bin_data.size**0.5 if bin_data.size else 0.
    x_out += dx / 2.
    y_out += dy / 2.
    x_out = x_out[:-1]
    y_out = y_out[:-1]
    if drop_nan:
        selection = (np.isnan(z_out) | np.isnan(z_err))
        z_out[selection] = 0
        z_err[selection] = 0
    return x_out, y_out, z_out, z_err