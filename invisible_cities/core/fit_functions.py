"""
A set of functions for data fitting.

GML November 2016
"""

import logging
import warnings
log = logging.getLogger(__name__)

import numpy   as np
import pandas  as pd
import inspect as insp
import scipy.optimize
import scipy.stats

from functools import wraps

from .                    import core_functions as coref
from .  stat_functions    import poisson_sigma
from .. evm.ic_containers import FitFunction
from .. database.load_db  import DB
from .. types.ic_types import FitPar
from .. types.ic_types import FitResult
from .. types.ic_types import NN
from .. core.core_functions import shift_to_bin_centers

from typing import Tuple


def fixed_parameters(fn, **kwargs):
    """
    Wrapper to fix parameters of a function
    which can then be used in curve_fit.
    """
    fn_sig    = insp.signature(fn)
    fn_pars   = np.array(list(fn_sig.parameters))[1:]
    par_gen   = (kwargs.get(k, np.nan) for k in fn_pars)
    all_args  = np.fromiter(par_gen, float)
    free_pars = np.isnan(all_args)

    if np.all(free_pars):
        raise ValueError(str(kwargs.keys()) + " not parameters " + fn.__name__)
    elif not np.any(free_pars):
        raise ValueError("Fixing all parameters pointless")
    elif np.count_nonzero(free_pars) > len(fn_pars) - len(kwargs):
        raise ValueError("Some parameters not found in " + fn.__name__)

    @wraps(fn)
    def fixed_fn(x, *pars):
        all_args[free_pars] = pars
        func_value = fn(x, *all_args)
        ## hack to maintain spe_functions underlying decorator
        if hasattr(fn, 'n_gaussians'):
            fixed_fn.n_gaussians = fn.n_gaussians
        return func_value

    ## Correct the doc string and signature for the new wrapped function
    fixed_fn.__doc__  = fixed_fn.__name__ + str(fixed_fn.__doc__) + "\n"
    fixed_fn.__doc__ += " fixed" + str(fn_pars[np.invert(free_pars)])

    new_pars               = np.array(tuple(fn_sig.parameters.values()))
    new_pars               = np.insert(new_pars[1:][free_pars], 0, new_pars[0])
    fixed_fn.__signature__ = fn_sig.replace(parameters=tuple(new_pars))
    return fixed_fn


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
def gauss(x, area, mu, sigma):
    """
    Gaussian function. Defined as A/sqrt(2 pi)/sigma exp(-(x-mu)**2/2/sigma**2),
    where A is the of the area (integral) of the function from -inf to +inf.
    The amplitude of the curve at x=mu is given by A/sqrt(2 pi)/sigma.
    """
    if sigma <= 0.:
        return np.inf
    return area/(2*np.pi)**.5/sigma * np.exp(-0.5*(x-mu)**2./sigma**2.)


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


def profile1d(z : np.array,
              e : np.array,
              nbins_z : int,
              range_z : np.array)->Tuple[float, float, float]:
    """Adds an extra layer to profileX, returning only valid points"""
    x, y, yu     = profileX(z, e, nbins_z, range_z)
    valid_points = ~np.isnan(yu)
    x    = x [valid_points]
    y    = y [valid_points]
    yu   = yu[valid_points]
    return x, y, yu


def sigmoid(x          : np.array,
            scale      : float,
            inflection : float,
            slope      : float,
            offset     : float)->np.array:

    return scale / ( 1 + np.exp( - slope * ( x - inflection ) ) ) + offset


def compute_drift_v(zdata    : np.array,
                    nbins    : int                               = 35,
                    zrange   : Tuple[float, float]               = (500, 640),
                    seed     : Tuple[float, float, float, float] = None,
                    detector : str                               = 'new')->Tuple[float, float]:
    """
    Computes the drift velocity for a given distribution
    using the sigmoid function to get the cathode edge.
    Parameters
    ----------
    zdata: array_like
        Values of Z coordinate.
    nbins: int (optional)
        The number of bins in the z coordinate for the binned fit.
    zrange: length-2 tuple (optional)
        Fix the range in z.
    seed: length-4 tuple (optional)
        Seed for the fit.
    detector: string (optional)
        Used to get the cathode position from DB.
    plot_fit: boolean (optional)
        Flag for plotting the results.
    Returns
    -------
    dv: float
        Drift velocity.
    dvu: float
        Drift velocity uncertainty.
    """

    y, x = np.histogram(zdata, nbins, zrange)
    x    = shift_to_bin_centers(x)

    if seed is None: seed = np.max(y), np.mean(zrange), 0.5, np.min(y)

    z_cathode = DB.DetectorGeo(detector).ZMAX[0]
    try:
        f = fit(sigmoid, x, y, seed, sigma=poisson_sigma(y), fit_range=zrange)
        dv  = z_cathode/f.values[1]
        dvu = dv / f.values[1] * f.errors[1]
    except RuntimeError:
        print("WARNING: Sigmoid fit for dv computation fails. NaN value will be set in its place.")
        dv, dvu = np.nan, np.nan

    return dv, dvu


def expo_seed(x, y, eps=1e-12):
    """
    Estimate the seed for a exponential fit to the input data.
    """
    x, y  = zip(*sorted(zip(x, y)))
    const = y[0]
    slope = (x[-1] - x[0]) / np.log(y[-1] / (y[0] + eps))
    seed  = const, slope
    return seed


def fit_lifetime_profile(z : np.array,
                         e : np.array,
                         nbins_z : int,
                         range_z : Tuple[float,float])->Tuple[FitPar, FitPar, FitResult]:
    """
    Make a profile of the input data and fit it to an exponential
    function.
    Parameters
    ----------
        z
            Array of z values.
        e
            Array of energy values.
        nbins_z
            Number of bins in Z for the profile fit.
        range_z
            Range in Z for fit.
    Returns
    -------
        A Tuple with:
            FitPar : Fit parameters (arrays of fitted values and errors, fit function)
            FitPar : Fit parameters (duplicated to make it compatible with fit_liftime_unbined)
            FirResults: Fit results (lt, e0, errors, chi2)
    @dataclass
    class ProfilePar:
        x  : np.array
        y  : np.array
        xu : np.array
        yu : np.array
    @dataclass
    class FitPar(ProfilePar):
        f     : FitFunction
    @dataclass
    class FitResult:
        par  : np.array
        err  : np.array
        chi2 : float
        valid : bool
    """

    logging.debug(' fit_liftime_profile')
    logging.debug(f' len (z) ={len(z)}, len (e) ={len(e)} ')
    logging.debug(f' nbins_z ={nbins_z}, range_z ={range_z} ')
    fp    = None
    valid = True
    c2    = NN
    par   = NN  * np.ones(2)
    err   = NN  * np.ones(2)

    x, y, yu  = profile1d(z, e, nbins_z, range_z)
    xu        = np.diff(x) * 0.5
    seed      = expo_seed(x, y)

    logging.debug(f' after profile: len (x) ={len(x)}, len (y) ={len(y)} ')
    try:
        f      = fit(expo, x, y, seed, sigma=yu)
        c2     = f.chi2
        par    = np.array(f.values)
        par[1] = - par[1]
        err    = np.array(f.errors)

        logging.debug(f' e0z ={par[0]} +- {err[0]} ')
        logging.debug(f' lt ={par[1]} +- {err[1]} ')
        logging.debug(f' c2 ={c2} ')
        fp = FitPar(x  = x,
                    y  = y,
                    xu = xu,
                    yu = yu,
                    f  = f.fn)
    except:
        warnings.warn(f' fit failed for seed  = {seed} in fit_lifetime_profile', UserWarning)
        valid = False
        raise

    fr = FitResult(par = par,
                   err = err,
                   chi2 = c2,
                   valid = valid)

    return fp, fp, fr


def gauss_seed(x, y, sigma_rel=0.05):
    """
    Estimate the seed for a gaussian fit to the input data.
    """
    y_max  = np.argmax(y) # highest bin
    x_max  = x[y_max]
    sigma  = sigma_rel * x_max
    amp    = y_max * (2 * np.pi)**0.5 * sigma * np.diff(x)[0]
    seed   = amp, x_max, sigma
    return seed


def quick_gauss_fit(data, bins):
    """
    Histogram input data and fit it to a gaussian with the parameters
    automatically estimated.
    """
    y, x  = np.histogram(data, bins)
    x     = shift_to_bin_centers(x)
    seed  = gauss_seed(x, y)
    f     = fit(gauss, x, y, seed)
    assert np.all(f.values != seed)
    return f