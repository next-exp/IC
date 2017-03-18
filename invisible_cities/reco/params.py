import sys
from collections import namedtuple

this_module = sys.modules[__name__]

def _add_namedtuple_in_this_module(name, attribute_names):
    new_nametuple = namedtuple(name, attribute_names)
    setattr(this_module, name, new_nametuple)

for name, attrs in (
        ('RawVectors'     , 'event pmtrwf sipmrwf pmt_active sipm_active'),
        ('SensorParams'   , 'NPMT PMTWL NSIPM SIPMWL'),
        ('CalibParams'    , 'coeff_c, coeff_blr, adc_to_pes_pmt adc_to_pes_sipm'),
        ('S12Params'      , 'tmin tmax stride lmin lmax rebin'),
        ('PmapParams'      ,'s1_params s2_params s1p_params s1_PMT_params s1p_PMT_params'),
        ('ThresholdParams', 'thr_s1 thr_s2 thr_MAU thr_sipm thr_SIPM'),
        ('CalibratedSum'  , 'csum csum_mau'),
        ('CalibratedPMT'  , 'CPMT CPMT_mau'),
        ('S1PMaps'        , 'S1 S1_PMT S1p S1p_PMT'),
        ('PMaps'          , 'S1 S2 S2Si'),
        ('Peak'           , 't E')):
    _add_namedtuple_in_this_module(name, attrs)

# Leave nothing but the namedtuple types in the namespace of this module
