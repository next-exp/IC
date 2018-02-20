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
         self.baselines) = DB.SiPMNoise(run_number)
        self.nsamples    = sample_size
        self.smear       = smear
        self.active      = DB.DataSiPM(run_number).Active.values[:, np.newaxis]
        self.adc_to_pes  = DB.DataSiPM(run_number).adc_to_pes.values.astype(np.double)[:, np.newaxis]
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
        find_thr = partial(inverse_cdf, self.xbins, percentile = noise_cut)
        cumprobs = np.apply_along_axis(np.cumsum, 1, self.probs)
        cuts_pes = np.apply_along_axis(find_thr , 1,   cumprobs)
        return cuts_pes * pes_to_adc
