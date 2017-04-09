import sys

import numpy as np

from collections import namedtuple


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
        ('Cluster'        , 'Q pos rms Nsipm'),
        ('Measurement'    , 'value uncertainty')):
    _add_namedtuple_in_this_module(name, attrs)

# Leave nothing but the namedtuple types in the namespace of this module
del name, namedtuple, sys, this_module, _add_namedtuple_in_this_module


class Correction:
    def __init__(self, xs, fs, us, norm = False):
        self.xs = np.array(xs, dtype=float, ndmin=2).T
        self.fs = np.array(fs, dtype=float)
        self.us = np.array(us, dtype=float)

        self._normalize(norm)

    def _normalize(self, norm):
        """
        Normalize. Options:
            - False:    Do not normalize.
            - "max":    Normalize to maximum energy encountered.
            - "center": Normalize to the energy placed at the center of the array.
        """
        if not norm:
            return
        elif norm == "max":
            index  = np.argmax(self.es)
        elif norm == "center":
            index  = [i//2 for i in self.es.shape]
        else:
            raise ValueError("Normalization option not recognized: {}".format(norm))

        f_norm  = self.fs[index]
        u_norm  = self.us[index]

        fs_norm = self.fs[index]/self.fs
        self.us = fs_norm * ((u_norm / f_norm)**2 +
                             (self.us/self.fs)**2 )**0.5
        self.fs = fs_norm

    def _find_closest(self, x):
        return np.apply_along_axis(np.argmin, 0, abs(x-self.xs))

    def __call__(self, *x):
        x_closest = np.apply_along_axis(self._find_closest, 1,
                                        np.array(x, ndmin=2))
        return self.fs[x_closest], self.us[x_closest]


