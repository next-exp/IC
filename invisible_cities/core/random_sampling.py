"""Defines a class for random sampling."""

import numpy as np

from functools import partial

from .. database import load_db as DB


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


class NoiseSampler:
    def __init__(self, run_number, sample_size=1, smear=True):
        """Sample a histogram as if it was a PDF.

        Parameters
        ----------
        run_number: int
            Run number used to load information from the database.
        sample_size: int
            Number of samples per sensor and call.
        smear: bool
            Flag to choose between performing discrete or continuous sampling.

        Attributes
        ---------
        baselines : array of floats
            Pedestal for each SiPM.
        xbins : numpy.ndarray
            Contains the the bins centers in pes.
        dx: float
            Half of the bin size.
        probs: numpy.ndarray
            Matrix holding the probability for each sensor at each bin.
        nsamples: int
            Number of samples per sensor taken at each call.
        """
        (self.probs,
         self.xbins,
         self.baselines) = DB.SiPMNoise()
        self.nsamples    = sample_size
        self.smear       = smear
        self.active      = DB.DataSiPM(run_number).Active.values[:, np.newaxis]
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
        """Return a sample of each distribution."""
        sample  = np.apply_along_axis(self._sampler, 1, self.probs)
        if self.smear:
            sample += self._smearer()
        sample += self.baselines
        return self.mask(sample)

    def compute_thresholds(self, noise_cut=0.99, pes_to_adc=None):
        """Find the number of pes at which each noise distribution leaves
        behind the a given fraction of its population.

        Parameters
        ----------
        noise_cut : float
            Fraction of the distribution to be left behind. Default is 0.99.
        pes_to_adc : float or array of floats, optional
            Constant(s) for pes to adc conversion (default None).
            If not present, the thresholds are given in pes.

        Returns
        -------
        cuts: array of floats
            Cuts in adc or pes.

        """
        if pes_to_adc is None:
            pes_to_adc = np.ones(self.probs.shape[0])

        find_thr = partial(inverse_cdf, self.xbins, percentile = noise_cut)
        cumprobs = np.apply_along_axis(np.cumsum, 1, self.probs)
        cuts_pes = np.apply_along_axis(find_thr , 1,   cumprobs)
        return cuts_pes * pes_to_adc
