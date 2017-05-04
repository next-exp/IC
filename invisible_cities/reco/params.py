import sys
from collections import namedtuple

import numpy as np


this_module = sys.modules[__name__]

def _add_namedtuple_in_this_module(name, attribute_names):
    new_nametuple = namedtuple(name, attribute_names)
    setattr(this_module, name, new_nametuple)

for name, attrs in (
        ('RawVectors'     , 'event pmtrwf sipmrwf pmt_active sipm_active'),
        ('SensorParams'   , 'NPMT PMTWL NSIPM SIPMWL'),
        ('CalibParams'    , 'coeff_c, coeff_blr, adc_to_pes_pmt adc_to_pes_sipm'),
        ('DeconvParams'   , 'n_baseline thr_trigger'),
        ('CalibVectors'   , 'channel_id coeff_blr coeff_c adc_to_pes adc_to_pes_sipm pmt_active'),
        ('S12Params'      , 'tmin tmax stride lmin lmax rebin'),
        ('PmapParams'     ,'s1_params s2_params s1p_params s1_PMT_params s1p_PMT_params'),
        ('ThresholdParams', 'thr_s1 thr_s2 thr_MAU thr_sipm thr_SIPM'),
        ('CalibratedSum'  , 'csum csum_mau'),
        ('CalibratedPMT'  , 'CPMT CPMT_mau'),
        ('S1PMaps'        , 'S1 S1_PMT S1p S1p_PMT'),
        ('PMaps'          , 'S1 S2 S2Si'),
        ('Peak'           , 't E'),
        ('FitFunction'    , 'fn values errors chi2 pvalue'),
        ('Cluster'        , 'Q pos rms Nsipm')):
    _add_namedtuple_in_this_module(name, attrs)

# Leave nothing but the namedtuple types in the namespace of this module
del name, namedtuple, sys, this_module, _add_namedtuple_in_this_module


class Correction:
    """
    Interface for accessing any kind of corrections.

    Parameters
    ----------
    xs : np.ndarray
        Array of coordinates corresponding to each correction.
    fs : np.ndarray
        Array of corrections or the values used for computing them.
    us : np.ndarray
        Array of uncertainties or the values used for computing them.
    norm : False or string
        Flag to set the normalization option. Options:
        - False:    Do not normalize.
        - "max":    Normalize to maximum energy encountered.
        - "center": Normalize to the energy placed at the center of the array.
    """
    def __init__(self, xs, fs, us, norm = False):
        self.xs = np.array(xs, dtype=float, ndmin=2).T
        self.fs = np.array(fs, dtype=float)
        self.us = np.array(us, dtype=float)

        self._normalize(norm)

    def _normalize(self, norm):
        if not norm:
            return
        elif norm == "max":
            index  = np.argmax(self.fs)
        elif norm == "center":
            # Take the index at the "center" of the array
            index  = tuple(i//2 for i in self.fs.shape)
        else:
            raise ValueError("Normalization option not recognized: {}".format(norm))

        # reference values
        f_norm  = self.fs[index]
        u_norm  = self.us[index]

        # Normalization is used only when the input is energy, which is used
        # to compute the correction factors.
        # These are meant to be used as multiplicative factors, so the
        # correction factor is reference energy/measured energy.
        fs_norm = self.fs[index]/self.fs

        # If the measured energy is zero, apply no correction at all.
        fs_norm[self.fs == 0] = 1

        # Propagate uncertainties
        self.us = fs_norm * ((u_norm / f_norm)**2 +
                             (self.us/self.fs)**2 )**0.5
        self.fs = fs_norm

    def _find_closest(self, x):
        # For each value in x, find the closest value in the bunch of coordinates
        return np.apply_along_axis(np.argmin, 0, abs(x-self.xs))

    def __call__(self, *x):
        """
        Compute the correction factor.
        
        Parameters
        ----------
        *x: Sequence of nd.arrays
             Each array is one coordinate. The number of coordinates must match
             that of the `xs` array in the init method.
        """
        # For each "event" in the input, find the closest point
        x_closest = np.apply_along_axis(self._find_closest, 0,
                                        np.array(x, ndmin=2))

        # List of tuples:
        # - tuple-index acts as a multiindex, i.e. one tuple = one element
        # - array-index acts as a multiindexer, i.e., one array = multiple elements
        #   which imay be confusing for multidimensional arrays.
        x_closest = list(map(tuple, x_closest))
        return self.fs[x_closest], self.us[x_closest]


