import os
import functools
import textwrap

import numpy             as np
import matplotlib.pyplot as plt

from .. core                import fit_functions as fitf
from .. core.core_functions import shift_to_bin_centers
from .. evm.ic_containers   import Measurement


def create_new_figure(kwargs):
    if kwargs.setdefault("new_figure", True):
        plt.figure()
    del kwargs["new_figure"]


def labels(xlabel, ylabel, title=""):
    """
    Set x and y labels.
    """
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title ( title)


def hbins(x, nsigma=5, nbins=10):
    """Given an array x, hbins returns the number of bins
    in an interval of  [<x> - nsigma*std(x), <x> + nsigma*std(x)]
    """
    xmin = np.average(x) - nsigma * np.std(x)
    xmax = np.average(x) + nsigma * np.std(x)
    bins = np.linspace(xmin, xmax, nbins + 1)
    return bins


def plot(*args, **kwargs):
    """
    Create a figure and then the histogram
    """
    create_new_figure(kwargs)
    return plt.plot(*args, **kwargs)


def hist(*args, **kwargs):
    """
    Create a figure and then the histogram
    """
    create_new_figure(kwargs)

    y, x, p = plt.hist(*args, **kwargs)
    return y, shift_to_bin_centers(x), p


def doublehist(data1, data2, lbls, *args, **kwargs):
    """
    Create a figure and then the histogram
    """
    create_new_figure(kwargs)

    h1 = hist(data1, *args, label=lbls[0], alpha=0.5, normed=True, new_figure=False, **kwargs)
    h2 = hist(data2, *args, label=lbls[1], alpha=0.5, normed=True, new_figure=False, **kwargs)
    return h1, h2, plt.legend()


def hist2d(*args, **kwargs):
    """
    Create a figure and then the histogram
    """
    create_new_figure(kwargs)

    z, x, y, p = plt.hist2d(*args, **kwargs)
    return z, shift_to_bin_centers(x), shift_to_bin_centers(y), p


def pdf(data, *args, **kwargs):
    """
    Create a figure and then the normalized histogram
    """
    create_new_figure(kwargs)

    h = hist(data, *args, **kwargs, weights=np.ones_like(data)/len(data))
    plt.yscale("log")
    plt.ylim(1e-4, 1.)
    return h


def scatter(*args, **kwargs):
    """
    Create a figure and then a scatter plot
    """
    create_new_figure(kwargs)
    return plt.scatter(*args, **kwargs)


def errorbar(*args, **kwargs):
    """
    Create a figure and then a scatter plot
    """
    create_new_figure(kwargs)
    return plt.errorbar(*args, **kwargs)


# I will leave this function here so old code does not crash,
# but the user will want to use the one after that
def profile_and_scatter(x, y, z, nbin, *args, **kwargs):
    """
    Create a figure and then a scatter plot
    """
    create_new_figure(kwargs)

    x, y, z, ze = fitf.profileXY(x, y, z, *nbin, *args, **kwargs)
    x_ = np.repeat(x, x.size)
    y_ = np.tile  (y, y.size)
    z_ = z.flatten()
    return (x, y, z, ze), plt.scatter(x_, y_, c=z_, marker="s"), plt.colorbar()


def hist2d_profile(x, y, z, nbinx, nbiny, xrange, yrange, **kwargs):
    """
    Create a profile 2d of the data and plot it as an histogram.
    """

    x, y, z, ze = fitf.profileXY(x, y, z, nbinx, nbiny, xrange, yrange)
    plot_output = display_matrix(x, y, z, **kwargs)
    return ((x, y, z, ze), *plot_output)


def display_matrix(x, y, z, mask=None, **kwargs):
    """
    Display the matrix z using the coordinates x and y as the bin centers.
    """
    nx = np.size(x)
    ny = np.size(y)

    dx = (np.max(x) - np.min(x)) / nx
    dy = (np.max(y) - np.min(y)) / ny

    x_binning = np.linspace(np.min(x) - dx, np.max(x) + dx, nx + 1)
    y_binning = np.linspace(np.min(y) - dy, np.max(y) + dy, ny + 1)

    x_ = np.repeat(x, ny)
    y_ = np.tile  (y, nx)
    z_ = z.flatten()

    if mask is None:
        mask = np.ones_like(z_, dtype=bool)
    else:
        mask = mask.flatten()
    h  = hist2d(x_[mask], y_[mask], (x_binning,
                                     y_binning),
                weights = z_[mask],
                **kwargs)
    return h, plt.colorbar()


def doublescatter(x1, y1, x2, y2, lbls, *args, **kwargs):
    """
    Create a figure and then a scatter plot
    """
    create_new_figure(kwargs)

    sc1 = scatter(x1, y1, *args, label=lbls[0], new_figure=False, **kwargs)
    sc2 = scatter(x2, y2, *args, label=lbls[1], new_figure=False, **kwargs)
    return sc1, sc2, plt.legend()


def covariance(x, y, **kwargs):
    """
    Display the eigenvectors of the covariance matrix.
    """
    create_new_figure(kwargs)

    cov = np.cov(x, y)
    l, v = np.linalg.eig(cov)
    lx, ly = l**0.5
    vx, vy = v.T
    x0, y0 = np.mean(x), np.mean(y)
    x1     = lx * vx[0]
    y1     = lx * vx[1]
    plt.arrow(x0, y0, x1, y1, head_width=0.1*ly, head_length=0.1*lx, fc='r', ec='r')
    x1     = ly * vy[0]
    y1     = ly * vy[1]
    plt.arrow(x0, y0, x1, y1, head_width=0.1*lx, head_length=0.1*ly, fc='r', ec='r')
    return l, v


def resolution(values, errors = None, E_from=41.5, E_to=2458):
    """
    Compute resolution at E_from and resolution at E_to
    with uncertainty propagation.
    """
    if errors is None:
        errors = np.zeros_like(values)

    amp  ,   mu,   sigma, *_ = values
    u_amp, u_mu, u_sigma, *_ = errors

    r   = 235. * sigma/mu
    u_r = r * (u_mu**2/mu**2 + u_sigma**2/sigma**2)**0.5

    scale = (E_from/E_to)**0.5
    return Measurement(r        , u_r        ), \
           Measurement(r * scale, u_r * scale)


def gausstext(values, errors, E_from=41.5, E_to=2458):
    """
    Build a string to be displayed within a matplotlib plot.
    """
    reso = resolution(values, errors, E_from=E_from, E_to=E_to)
    E_to = "Qbb" if E_to==2458 else str(E_to) + " keV"
    return textwrap.dedent("""
        $\mu$ = {0}
        $\sigma$ = {1}
        R = {2} % @ {3} keV
        R = {4} % @ {5}""".format(measurement_string(values[1] , errors[1]),
                                  measurement_string(values[2] , errors[2]),
                                  measurement_string(* reso[0]), E_from,
                                  measurement_string(* reso[1]), E_to))


def save_to_folder(outputfolder, name, format="png", dpi=100):
    """
    Set title and save plot in folder.
    """
    plt.savefig("{}/{}.{}".format(outputfolder, name, format), dpi=dpi)


def plot_writer(outputfolder, format, dpi=100):
    """
    Build a partial implementation of the save_to_folder function ensuring
    the output folder exists.
    """
    os.makedirs(outputfolder,  exist_ok = True)

    return functools.partial(save_to_folder,
                             outputfolder,
                             format = format,
                             dpi    = dpi)


def measurement_string(x, u_x):
    """
    Display a value-uncertainty pair with the same precision.
    """
    scale = int(np.floor(np.log10(u_x)))
    if scale >= 2:
        return "({}) · 1e{}".format(measurement_string(  x/10**scale,
                                                       u_x/10**scale),
                                    scale)
    n = 1 - scale

    format = "{" + ":.{}f".format(n) + "}"
    string = "{} +- {}".format(format, format)

    return string.format(np.round(  x, n),
                         np.round(u_x, n))
