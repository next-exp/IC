"""Defines a class for random sampling."""

import numpy as np

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
        def norm(ps):
            return ps / np.sum(ps) if ps.any() else ps

        self.nsamples    = sample_size
        (self.probs,
         self.xbins,
         self.baselines) = DB.SiPMNoise()
        self.active      = DB.DataSiPM(run_number).Active.values[:, np.newaxis]

        self.probs       = np.apply_along_axis(norm, 1, self.mask(self.probs))
        self.baselines   = self.baselines[:, np.newaxis]
        self.dx          = np.diff(self.xbins)[0] * 0.5

        # Sampling functions
        def _sample_sensor(probs):
            if not probs.any():
                return np.zeros(self.nsamples)
            return np.random.choice(self.xbins, size=self.nsamples, p=probs)

        def _discrete_sampler():
            return np.apply_along_axis(_sample_sensor, 1, self.probs)

        def _continuous_sampler():
            return _discrete_sampler() + np.random.uniform(-self.dx, self.dx)

        self._sampler = _continuous_sampler if smear else _discrete_sampler

    def mask(self, array):
        """Set to 0 those rows corresponding to masked sensors"""
        return array * self.active

    def sample(self):
        """Return a sample of each distribution."""
        return self.mask(self._sampler() + self.baselines)

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
        def findcut(probs):
            if not probs.any():
                return np.inf
            return self.xbins[probs >= noise_cut][0]

        if pes_to_adc is None:
            pes_to_adc = np.ones(self.probs.shape[0])

        cumprobs = np.apply_along_axis(np.cumsum, 1, self.probs)
        cuts_pes = np.apply_along_axis(findcut  , 1,   cumprobs)
        return cuts_pes * pes_to_adc
