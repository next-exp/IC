from enum import Enum

import numpy as np

from functools import partial
from functools import lru_cache

from .. database import load_db as DB

class DarkModel(Enum):
    mean      = 0
    threshold = 1

def normalize_distribution(y):
    ysum = np.sum(y)
    return y / ysum if ysum else y


def sample_discrete_distribution(x, y, size=1):
    if not y.any():
        return np.zeros(size)
    return np.random.choice(x, p=y, size=size)


def uniform_smearing(max_deviation, size=1):
    return np.random.uniform(-max_deviation,
                             +max_deviation,
                             size = size)


def inverse_cdf_index(y, percentile):
    return np.argwhere(y >= percentile)[0,0]


def inverse_cdf(x, y, percentile):
    if not y.any():
        return np.inf
    return x[inverse_cdf_index(y, percentile)]


def pad_pdfs(bins, spectra=None):
    bin_diff = np.count_nonzero(bins <= 0) - np.count_nonzero(bins > 0)
    pad = abs(min(bin_diff, 0)), abs(max(bin_diff, 0))
    if spectra is None:
        ## We just want to extend the bins
        return np.pad(bins, pad, 'linear_ramp',
                      end_values=(-bins[-1], -bins[0]))

    return np.apply_along_axis(np.pad, 1, spectra, pad,
                               'constant', constant_values=0)


def general_thresholds(xbins, probs, noise_cut):
    """
    Generalised version of NoiseSampler
    which can use any distribution.

    Parameters
    ----------
    xbins : array of floats
        The positions of the bin centres in x
    probs : array of floats
        Probabilities for each of the x bins
    noise_cut : float
        Fraction of the distribution to be left behind. Default is 0.99.

    Returns
    -------
    cuts: array of floats
        Cuts in adc or pes.
    """
    find_thr = partial(inverse_cdf, xbins, percentile = noise_cut)
    cumprobs = np.apply_along_axis(np.cumsum, 1, probs)
    cuts     = np.apply_along_axis(find_thr , 1, cumprobs)
    return cuts


class NoiseSampler:
    def __init__(self, detector, run_number, sample_size=1, smear=True):
        """Sample a histogram as if it was a PDF.

        Parameters
        ----------
        run_number: int
            Run number used to load information from the database.
        sample_size: int
            Number of samples per sensor and call.
        smear: bool, optional
            If True, the samples are uniformly smeared to simulate
            a continuous distribution. If False, the samples are
            always the center of the histograms' bins. Default is True.

        Attributes
        ---------
        baselines : array of floats
            Baseline for each SiPM.
        xbins : numpy.ndarray
            Contains the the bins centers in pes.
        dx : float
            Half of the bin size.
        probs: numpy.ndarray
            Matrix holding the noise probabilities for each sensor.
            The sensors are arranged along the first dimension, while
            the other axis corresponds to the energy bins.
        """
        (self.probs,
         self.xbins,
         self.baselines) = DB.SiPMNoise(detector, run_number)
        self.nsamples    = sample_size
        self.smear       = smear
        self.active      = DB.DataSiPM(detector, run_number).Active.values[:, np.newaxis]
        self.adc_to_pes  = DB.DataSiPM(detector, run_number).adc_to_pes.values.astype(np.double)[:, np.newaxis]
        self.nsensors    = self.active.size

        self.probs       = np.apply_along_axis(normalize_distribution, 1,
                                               self.mask(self.probs))
        self.baselines   = self.baselines[:, np.newaxis]
        self.dx          = np.diff(self.xbins)[0] * 0.5

        self._sampler    = partial(sample_discrete_distribution,
                                   self.xbins,
                                   size = self.nsamples)
        self._smearer    = partial(uniform_smearing,
                                   max_deviation = self.dx,
                                   size          = (self.nsensors,
                                                    self.nsamples))

    def mask(self, array):
        """Set to 0 those rows corresponding to masked sensors"""
        return array * self.active

    def sample(self):
        """Take a set of samples from each pdf."""
        sample  = np.apply_along_axis(self._sampler, 1, self.probs)
        if self.smear:
            sample += self._smearer()
        sample = self.adc_to_pes * sample + self.baselines
        return self.mask(sample)

    def compute_thresholds(self, noise_cut=0.99, pes_to_adc=1):
        """Find the energy threshold that reduces the noise population by a
        fraction of *noise_cut*.

        Parameters
        ----------
        noise_cut : float, optional
            Fraction of the distribution to be left behind. Default is 0.99.
        pes_to_adc : float or array of floats, optional
            Constant(s) for pes to adc conversion (default None).
            If not present, the thresholds are given in pes.

        Returns
        -------
        cuts: array of floats
            Cuts in adc or pes.
        """
        cuts_pes = general_thresholds(self.xbins, self.probs, noise_cut)
        return cuts_pes * pes_to_adc


    def signal_to_noise(self, ids, charges,
                        sample_width, dark_model=DarkModel.threshold):
        """
        Find the signal to noise for the sipms in the array

        Parameters
        ----------
        ids : array of ints
            Array of all sipm ids with charge in the slice.
        charges : array of floats
            Array of charge seen by all sipm which see light
            in the slice.
        sample_width : int
            The width in mus of the slice
        dark_model : Enum
            The model for dark counts, mean or threshold

        Returns
        -------
        signal_to_noise : array of floats
            An array of S/N values for the sipms in the slice.
        """

        dark_levels = self.dark_expectation(sample_width, dark_model)

        return charges / np.sqrt(charges + dark_levels[ids])


    @lru_cache(maxsize=30)
    def dark_expectation(self, sample_width,
                         dark_model=DarkModel.threshold):
        """
        Calculate, for a given sample_width,
        the mean expectation to approximate
        dark counts for all sipm channels.
        """
        pdfs = self.multi_sample_distributions(sample_width)

        pad_xbins = pad_pdfs(self.xbins)

        if dark_model == DarkModel.threshold:
            dark_pes = general_thresholds(pad_xbins, pdfs, 0.99)
            dark_pes[self.active.flatten() == 0] = 0
            return dark_pes
        
        active           = self.active.flatten() != 0
        nsipm            = np.count_nonzero(active)
        dark_estimate    = np.average(np.repeat(pad_xbins[None, :], nsipm,  0),
                                      axis = 1,         weights = pdfs[active])
        
        dark_pes         = np.full(self.nsensors, 0, dtype=np.float)
        dark_pes[active] = dark_estimate
        return dark_pes


    @lru_cache(maxsize=30)
    def multi_sample_distributions(self, sample_width):
        """
        Calculate the no light PDFs for a given
        sample width.
        Parameters
        __________
        sample_width : int
            Width in microseconds of the sample for which
            PDFs to be calculated.

        Returns
        _______
        PDFs : numpy.ndarray shape (nsipm, xbins + padding)
            Probabilities for each xbin recalculated for
            the requested sample_width and padded for symmetry
            around zero.
        """
        if sample_width == 1:
            return pad_pdfs(self.xbins, self.probs)

        mapping = map(np.convolve                                      ,
                      self.multi_sample_distributions(               1),
                      self.multi_sample_distributions(sample_width - 1),
                      np.full(self.probs.shape[0], "same")             )

        return np.array(tuple(mapping))
