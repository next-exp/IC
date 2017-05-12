"""Defines a class for random sampling."""

import numpy as np

from .. database import load_db as DB


class NoiseSampler:
    def __init__(self, sample_size=1, smear=True):
        """Sample a histogram as if it was a PDF.

        Parameters
        ----------
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

        self.nsamples = sample_size
        self.probs, self.xbins, self.baselines = DB.SiPMNoise()

        self.probs = np.apply_along_axis(norm, 1, self.probs)
        self.baselines = self.baselines.reshape(self.baselines.shape[0], 1)
        self.dx = np.diff(self.xbins)[0] * 0.5

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

    def Sample(self):
        """Return a sample of each distribution."""
        return self._sampler() + self.baselines

    def ComputeThresholds(self, noise_cut=0.99, pes_to_adc=None):
        """Find the number of pes at which each noise distribution leaves
        behind the a given fraction of its population.

        Parameters
        ----------
        noise_cut : float
            Fraction of the distribution to be left behind. Default is 0.99.
        pes_to_adc : float or array of floats, optional
            Constant(s) for adc to pes conversion (default None).
            If not present, the thresholds are given in pes.

        Returns
        -------
        cuts: array of floats
            Cuts in adc or pes.

        """
        def findcut(probs):
            if not probs.any():
                return np.inf
            return self.xbins[probs > noise_cut][0]

        if pes_to_adc is None:
            pes_to_adc = np.ones(self.probs.shape[0])
        pes_to_adc.reshape(self.probs.shape[0], 1)

        cumprobs = np.apply_along_axis(np.cumsum, 1, self.probs)
        return np.apply_along_axis(findcut, 1, cumprobs) * pes_to_adc
