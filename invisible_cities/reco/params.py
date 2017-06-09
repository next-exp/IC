import sys

from collections import namedtuple


this_module = sys.modules[__name__]

def _add_namedtuple_in_this_module(name, attribute_names):
    new_nametuple = namedtuple(name, attribute_names)
    setattr(this_module, name, new_nametuple)


# TODO: move this into a more appropriate module
class minmax:

    def __init__(self, min, max):
        assert min <= max
        self.min = min
        self.max = max

    def __mul__(self, factor):
        return minmax(self.min * factor, self.max * factor)

    def __add__(self, scalar):
        return minmax(self.min + scalar, self.max + scalar)

    def __sub__(self, scalar):
        return minmax(self.min - scalar, self.max - scalar)

    def __str__(self):
        return 'minmax(min={.min}, max={.max})'.format(self, self)
    __repr__ = __str__

    def __getitem__(self, n):
        if n == 0: return self.min
        if n == 1: return self.max
        raise IndexError


for name, attrs in (
        ('RawVectors'     , 'event pmtrwf sipmrwf pmt_active sipm_active'),
        ('SensorParams'   , 'NPMT PMTWL NSIPM SIPMWL'),
        ('CalibParams'    , 'coeff_c, coeff_blr, adc_to_pes_pmt adc_to_pes_sipm'),
        ('DeconvParams'   , 'n_baseline thr_trigger'),
        ('CalibVectors'   , 'channel_id coeff_blr coeff_c adc_to_pes adc_to_pes_sipm pmt_active'),
        ('S12Params'      , 'time stride length rebin'),
        ('PmapParams'     , 's1_params s2_params s1p_params s1_PMT_params s1p_PMT_params'),
        ('ThresholdParams', 'thr_s1 thr_s2 thr_MAU thr_sipm thr_SIPM'),
        ('CalibratedSum'  , 'csum csum_mau'),
        ('CalibratedPMT'  , 'CPMT CPMT_mau'),
        ('S1PMaps'        , 'S1 S1_PMT S1p S1p_PMT'),
        ('PMaps'          , 'S1 S2 S2Si'),
        ('Peak'           , 't E'),
        ('FitFunction'    , 'fn values errors chi2 pvalue'),
        ('Cluster'        , 'Q pos rms nsipm'),
        ('TriggerParams'  , 'trigger_channels min_number_channels charge height width'),
        ('PeakData'       , 'charge height width'),
        ('Measurement'    , 'value uncertainty')):
    _add_namedtuple_in_this_module(name, attrs)

# Leave nothing but the namedtuple types in the namespace of this module
del name, namedtuple, sys, this_module, _add_namedtuple_in_this_module
