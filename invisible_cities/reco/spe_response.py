from itertools import count
from functools import wraps

import numpy as np

from .. core .stat_functions import poisson_factor
from .. core . fit_functions import gauss


def get_padding(bins):
    """
    Get number of zeros to be padded at each end of an array
    to make it symmetric around 0.
    """
    bin_diff = np.count_nonzero(bins <= 0) - np.count_nonzero(bins > 0)
    return abs(min(bin_diff, 0)), abs(max(bin_diff, 0))


def number_of_gaussians(min_integral, scale, poisson_mean):
    """
    Count number of poisson-scaled gaussians which integral
    exceed a minimum value.
    """
    integrals = []
    for n_gaussians in count():
        integral = scale * poisson_factor(n_gaussians, poisson_mean)
        if integral <= min_integral:
            return n_gaussians, tuple(integrals)
        integrals.append(integral)


def suppress_negative_energy_contribution(xs, ys):
    """
    Set values in `ys` to zero when `xs` is below zero.
    A photon detection cannot create negative signal.
    """
    return np.where(xs >= 0, ys, 0)


def binned_gaussian_spectrum(centroid, sigma, integral, bins):
    samples     = np.random.normal(centroid, sigma, size=integral)
    spectrum, _ = np.histogram(samples, bins)
    return spectrum


# TODO: IMPROVE THIS BOTCHED JOB!!!
def set_n_gaussians(function):
    @wraps(function)
    def _function(*args, **kwargs):
        result,   n_gaussians = function(*args, **kwargs)
        _function.n_gaussians = n_gaussians
        return result
    return _function


def poisson_scaled_gaussians(                  *,
                             first        = 0   ,
                             n_gaussians  = None,
                             min_integral = None):
    """
    Produces the function that generates the spectrum
    resulting from adding a number of gaussians that
    are scaled according a poisson distribution. The
    pedestal is also modeled with a gaussian function.

    Keyword-only parameters
    -----------------------
    first : integer
        First gaussian to be taken into account
    n_gaussians : integer or tuple
        Number of gaussians to be added or first and last
        gaussian to be taken into account. (See notes)
    min_integral : float
        Minimum integral for a given gaussian to be considered. (See notes)

    Notes
    -----
    Either `n_gaussians` or `min_integral` must be given.

    Returns
    -------
    sum_of_gaussians: function
        Energy spectrum function.

    Raises
    ------
    ValueError: in case neither or both `n_gaussians` and
                `min_integral` are given.
    """
    if n_gaussians is None and min_integral is None:
        raise ValueError(     "Either n_gaussians or min_integral must be given")

    if min_integral is not None and n_gaussians is not None:
        raise ValueError("Only one of n_gaussians or min_integral must be given")

    @set_n_gaussians
    def sum_of_gaussians(xs,
                         scale, poisson_mean,
                         pedestal_mean, pedestal_sigma,
                         gain, gain_sigma):
        """
        Parameters
        ----------
        xs : np.ndarray with shape (n,)
            Energies at which the spectrum is evaluated.
        scale : float
            Overall scale factor
        poisson_mean : float
            Average probability of a photon being detected.
        pedestal_mean : float
            Baseline.
        pedestal_sigma : float
            Average noise.
        gain: float
            The sensor's gain, i.e. the average number of
            adc counts associated to a photon detection.
        gain_sigma : float
            Gain's standard deviation.

        Returns
        -------
        summed_spectrum: np.ndarray with shape (n,)
            Energy spectrum.
        """
        if min_integral is None:
            ngaussians = n_gaussians
            integrals  = tuple(scale * poisson_factor(k, poisson_mean) for k in range(ngaussians))
        else:
            (ngaussians,
             integrals ) = number_of_gaussians(min_integral, scale, poisson_mean)

        def spectrum_component(i):
            centroid       =  pedestal_mean     + i * gain
            sigma          = (pedestal_sigma**2 + i * gain_sigma**2)**0.5
            gauss_spectrum = gauss(xs, 1, centroid, sigma)
            if i: gauss_spectrum = suppress_negative_energy_contribution(xs, gauss_spectrum)

            return integrals[i] * gauss_spectrum

        return sum(map(spectrum_component, range(first, ngaussians))), ngaussians

    return sum_of_gaussians


def scaled_dark_pedestal(dark_spectrum,
                         pedestal_mean, pedestal_sigma,
                         min_integral):
    """
    Produces the function that generates the spectrum
    resulting from adding a number of gaussians that
    are scaled according a poisson distribution. The
    pedestal is modeled with a histogram got from
    sampling a gaussian distribution.

    Parameters
    ----------
    bins: np.ndarray with shape (n,)
        Histogram binning.
    nsamples: int
        Number of samples to be drawn to model the pedestal.
    pedestal_mean: float
        Baseline.
    pedestal_sigma: float
        Average noise.
    min_integral: float
        Minimum integral for a given gaussian to be considered.

    Returns
    -------
    scaled_dark_pedestal: function
        Energy spectrum function.
    """
    sum_of_gaussians = poisson_scaled_gaussians(first=1, min_integral=min_integral)

    @set_n_gaussians
    def scaled_dark_pedestal(xs,
                             scale, poisson_mean,
                             gain, gain_sigma):
        """
        Parameters
        ----------
        xs : np.ndarray with shape (n,)
            Energies at which the spectrum is evaluated.
        scale : float
            Overall scale factor
        poisson_mean : float
            Average probability of a photon being detected.
        gain: float
            The sensor's gain, i.e. the average number of
            adc counts associated to a photon detection.
        gain_sigma : float
            Gain's standard deviation.

        Returns
        -------
        summed_spectrum: np.ndarray with shape (n,)
            Energy spectrum.
        """
        pedestal = np.exp(-poisson_mean) * dark_spectrum
        signal   = sum_of_gaussians(xs,
                                    scale, poisson_mean,
                                    pedestal_mean, pedestal_sigma,
                                    gain, gain_sigma)
        return pedestal + signal, sum_of_gaussians.n_gaussians
    return scaled_dark_pedestal


def dark_convolution(bins, dark_spectrum,
                     min_integral):
    """
    Convolution of dark spectrum with gaussians.
    Produces the function that generates the spectrum
    resulting from convolving a number of gaussians that
    are scaled according a poisson distribution with
    a pedestal that is modeled with a histogram got from
    sampling a gaussian distribution.

    Parameters
    ----------
    bins: np.ndarray with shape (n,)
        Histogram binning.
    nsamples: int
        Number of samples to be drawn to model the pedestal.
    pedestal_mean: float
        Baseline.
    pedestal_sigma: float
        Average noise.
    min_integral: float
        Minimum integral for a given gaussian to be considered.

    Returns
    -------
    dark_convolution: function
        Energy spectrum function.
    """
    pad       = get_padding(bins)
    dark_norm = dark_spectrum / dark_spectrum.sum()
    dark_norm = np.pad(dark_norm, pad, "constant", constant_values=0)

    @set_n_gaussians
    def dark_convolution(xs,
                         scale, poisson_mean,
                         gain, gain_sigma):
        """
        Parameters
        ----------
        xs : np.ndarray with shape (n,)
            Energies at which the spectrum is evaluated.
        scale : float
            Overall scale factor
        poisson_mean : float
            Average probability of a photon being detected.
        gain: float
            The sensor's gain, i.e. the average number of
            adc counts associated to a photon detection.
        gain_sigma : float
            Gain's standard deviation.

        Returns
        -------
        summed_spectrum: np.ndarray with shape (n,)
            Energy spectrum.
        """
        pedestal    = np.exp(-poisson_mean) * dark_spectrum
        pe_gaussian = gauss(xs, 1, gain, gain_sigma)
        pe_gaussian = suppress_negative_energy_contribution(xs, pe_gaussian)
        pe_gaussian = np.pad(pe_gaussian, pad, "constant", constant_values=0)

        (n_gaussians,
         integrals  ) = number_of_gaussians(min_integral, scale, poisson_mean)
        signal        = np.zeros_like(pedestal)
        convolved     = dark_norm
        slice_        = slice(pad[0], len(convolved) - pad[1])
        for i in range(1, n_gaussians):
            convolved  = np.convolve(convolved, pe_gaussian, "same")
            signal    += integrals[i] * convolved[slice_]

        return pedestal + signal, n_gaussians
    return dark_convolution
