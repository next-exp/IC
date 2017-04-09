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
    def __init__(self, xs, es, ss, norm_opt = "max"):
        self.xs = np.array(xs, dtype=float, ndmin=2).T
        self.es = np.array(es, dtype=float)
        self.ss = np.array(es, dtype=float)

        self.e_norm, self.s_norm = \
        self._find_norm_values(norm_opt)

        self.s0 = (self.s_norm/self.e_norm)**2

    def _find_norm_values(self, norm_opt):
        if   norm_opt == "max": # normalize to max energy
            index  = np.argmax(self.es)
        elif norm_opt == "center":
            index  = [i//2 for i in self.es.shape]
        else:
            raise ValueError("Normalization option not recognized: {}".format(norm_opt))
        return self.es[index], self.ss[index]

    def _find_closest(self, x):
        return np.apply_along_axis(np.argmin, 0, abs(x-self.xs))

    def __call__(self, *x):
        x_closest   = np.apply_along_axis(self._find_closest, 1, np.array(x, ndmin=2))
        e_closest   = self.es[x_closest]
        s_closest   = self.ss[x_closest]
        correction  = self.e_norm/e_closest
        uncertainty = correction * (self.s0 + (s_closest/e_closest)**2)**0.5
        return correction, uncertainty


