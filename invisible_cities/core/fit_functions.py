"""
A set of functions for data fitting.

GML November 2016
"""

import numpy as np
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

## Sensor response function as class
class sensorSpeResponse:
    """
    Class for fitting sensor single photon response.
    Contains various functions to be able to fit under
    different conditions:
    """
    
    def __init__(self, bins, integ, cent=0., sig=1.):
        ## Initialize with a Gaussian
        self.dspec = np.histogram(np.random.normal(cent, sig, integ), bins)[0]

        ## Possible fixed noise position and sigma
        self.x0 = cent
        self.s0 = sig

        ## Initialize a Gaussian counter (default value)
        self.nGau = 7

        ## Default value for minimum integral
        ## for Gaussians.
        self.min_integ = 100.

        ## In case we want to pad to do a convolution
        # difference bins before and after 0
        bin_diff = len(np.argwhere(bins<=0)) - len(np.argwhere(bins>0))
        self.pad = (abs(min(bin_diff, 0)), abs(max(bin_diff,0)))

    ## Possibility to set a spectrum as 'dark'
    def set_dark_func(self, dark, cent=0., sig=1., scale=1.):
        """
        dark  : np array representing no light spectrum
        cent  : centroid to act in certain functions as 0
        sig   : rms of pedestal
        scale : additional scaling factor in case dark and
        led spectra don't have same number of entries.
        """
        self.dspec = dark * scale

        self.x0 = cent
        self.s0 = sig

    ## In case want to change binning and need to repad.
    ## Only needed for convolution function
    def redefine_bins(self, bins):

        bin_diff = len(np.argwhere(bins<=0)) - len(np.argwhere(bins>0))
        self.pad = (abs(min(bin_diff, 0)), abs(max(bin_diff,0)))

    ## Assumes Gaussian for pedestal and a set number of
    ## Gaussians for 1 -> nGau pe
    def set_gaussians(self, xs, *ps):
        """
        ps[0] - scale
        ps[1] - origin (usually 0.)
        ps[2] - period
        ps[3] - poisson mean (dark current)
        ps[4] - noise (peak 0)
        ps[5] - sigma of the first gaussian
        """
        ifacto = np.array(map(math.factorial, range(self.nGau)))
        efacto = 1./math.sqrt(2.*math.pi)
        nn, x0, pe, mu, s0, s1 = ps

        def ifun_(i):
            s2 = s0*s0 + i*s1*s1
            xref = x0+i*pe
            fact = (efacto/math.sqrt(s2))
            yg = fact*np.exp(-(xs-xref)*(xs-xref)/(2.*s2))
            yp = (math.exp(i*math.log(mu))/ifacto[i])
            return yg*yp

        iys = map(ifun_, range(ngauss))
        ys = reduce(lambda x, y: x+y, iys)
        ys = nn*math.exp(-mu)*ys
        return ys

    ## As above but fitting Gaussians up to min_integ
    def min_integ_gaussians(self, xs, *ps):
        """ function to fo a poisson distribution with gaussian peaks
        ps[0] - scale
        ps[1] - origin (usually 0.)
        ps[2] - 1pe mean
        ps[3] - poisson mean (dark current)
        ps[4] - noise (peak 0)
        ps[5] - sigma of the first gaussian
        min_integ controles the maximum number of Gaussians to calculate
        """
        efacto = 1./np.sqrt(2.*np.pi)
        nn, x0, pe, mu, s0, s1 = ps

        def ifun_():
            self.nGau = 0
            norm = nn*mu**self.nGau*np.exp(-mu)/np.math.factorial(self.nGau)
            while norm >= self.min_integ:
                s2 = s0*s0 + self.nGau*s1*s1
                xref = x0 + self.nGau*pe
                fact = (efacto/np.sqrt(s2))
                yg = fact*np.exp(-(xs-xref)*(xs-xref)/(2.*s2))
                # pe's can't really contribute to -ves
                yg[xs<0] = 0
                yield norm*yg
                self.nGau += 1
                norm *= mu/self.nGau

        ys = np.sum(ifun_())
        return ys

    ## Scaled dark spectrum for no light distribution
    def scaled_dark_pedestal(self, xs, *ps):
        """ 
        ps[0] - scale
        ps[1] - 1pe mean
        ps[2] - poisson mean (dark current)
        ps[3] - sigma of the first gaussian
        min_integ controles the maximum number of Gaussians to calculate
        """
        efacto = 1./np.sqrt(2.*np.pi)
        nn, pe, mu, s1 = ps

        ped_spec = np.exp(-mu) * self.dspec
        def ifun_():
            self.nGau = 1
            norm = nn*mu**self.nGau*np.exp(-mu)/np.math.factorial(self.nGau)
            while norm >= self.min_integ:
                s2 = self.s0*self.s0 + self.nGau*s1*s1
                xref = self.x0 + self.nGau*pe
                fact = (efacto/np.sqrt(s2))
                yg = fact*np.exp(-(xs-xref)*(xs-xref)/(2.*s2))
                # pe's can't really contribute to -ves
                yg[xs<0] = 0
                yield norm*yg
                self.nGau += 1
                norm *= mu/self.nGau

        ys = ped_spec + np.sum(ifun_())
        return ys

    ## Convolution of dark spectrum with Gaussians
    def dark_convolution(self, xs, *ps):
        """ response function convoluting a dark spectrum
        with multiple Gaussian distributions.
        ps[0] - scale
        ps[1] - 1pe mean
        ps[2] - poisson mean (dark current)
        ps[3] - sigma of the first gaussian
        min_integ controles the maximum number of Gaussians to calculate
        """
        nn, pe, mu, s1 = ps

        g_norm = 1./(np.sqrt(2.*np.pi) * s1)
        ped_spec = np.exp(-mu) * self.dspec
        dnorm = np.pad(self.dspec/self.dspec.sum(), self.pad,
                       'constant', constant_values=0.)

        ## FFTs of the base function: 1pe
        ## Single photon
        speG = g_norm * np.exp(-(xs-pe)*(xs-pe)/(2.*s1**2))
        speG[xs<0] = 0
        speG = np.pad(speG, self.pad, 'constant', constant_values=0.)
        ##
        def ifun_():
            self.nGau = 1
            norm = nn*mu**self.nGau*np.exp(-mu)/np.math.factorial(self.nGau)
            conv = np.convolve(dnorm, speG, 'same')
            while norm >= self.min_integ:
                if self.nGau > 1:
                    conv = np.convolve(conv, speG, 'same')
                yield norm * conv[self.pad[0]:len(conv)-self.pad[-1]]
                self.nGau += 1
                norm *= mu/self.nGau

        ys = ped_spec + np.sum(ifun_())
        return ys
    

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


    return FitFunction(fitf, vals, errors, chi2, pval)


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
    xmin, xmax = xrange if xrange else (np.min(xdata), np.max(xdata))
    ymin, ymax = yrange if yrange else (np.min(ydata), np.max(ydata))

    x_out = np.linspace(xmin, xmax, nbins+1)
    y_out = np.empty(nbins)
    y_err = np.empty(nbins)
    dx    = x_out[1] - x_out[0]

    selection = (coref.in_range(xdata, xmin, xmax) &
                 coref.in_range(ydata, ymin, ymax))
    x, y = xdata[selection], ydata[selection]
    for i in range(nbins):
        bin_data = y[coref.in_range(x,
                                    minval = x_out[i],
                                    maxval = x_out[i+1])]
        y_out[i] = coref.mean_handle_empty(bin_data)
        y_err[i] = coref. std_handle_empty(bin_data)
        if not std:
            y_err[i] /= bin_data.size ** 0.5

    x_out += dx / 2.
    x_out  = x_out[:-1]
    if drop_nan:
        selection = ~(np.isnan(y_out) | np.isnan(y_err))
        x_out = x_out[selection]
        y_out = y_out[selection]
        y_err = y_err[selection]
    return x_out, y_out, y_err


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
    xmin, xmax = xrange if xrange else (np.min(xdata), np.max(xdata))
    ymin, ymax = yrange if yrange else (np.min(ydata), np.max(ydata))
    zmin, zmax = zrange if zrange else (np.min(zdata), np.max(zdata))

    x_out = np.linspace(xmin, xmax, nbinsx+1)
    y_out = np.linspace(ymin, ymax, nbinsy+1)
    z_out = np.empty((nbinsx, nbinsy))
    z_err = np.empty((nbinsx, nbinsy))
    dx = x_out[1] - x_out[0]
    dy = y_out[1] - y_out[0]

    selection = (coref.in_range(xdata, xmin, xmax) &
                 coref.in_range(ydata, ymin, ymax) &
                 coref.in_range(zdata, zmin, zmax))
    xdata, ydata, zdata = xdata[selection], ydata[selection], zdata[selection]
    for i in range(nbinsx):
        for j in range(nbinsy):
            selection = (coref.in_range(xdata, x_out[i], x_out[i+1]) &
                         coref.in_range(ydata, y_out[j], y_out[j+1]))
            bin_data = zdata[selection]
            z_out[i,j] = coref.mean_handle_empty(bin_data)
            z_err[i,j] = coref. std_handle_empty(bin_data)
            if not std:
                z_err[i,j] /= bin_data.size ** 0.5
    x_out += dx / 2.
    y_out += dy / 2.
    x_out  = x_out[:-1]
    y_out  = y_out[:-1]
    if drop_nan:
        selection = (np.isnan(z_out) | np.isnan(z_err))
        z_out[selection] = 0
        z_err[selection] = 0
    return x_out, y_out, z_out, z_err
