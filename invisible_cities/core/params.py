import sys
from collections import namedtuple

this_module = sys.modules[__name__]

def _add_namedtuple_in_this_module(name, attribute_names):
    new_nametuple = namedtuple(name, attribute_names)
    setattr(this_module, name, new_nametuple)

for name, attrs in (
        ('S12Params'      , 'tmin tmax stride lmin lmax rebin'),
        ('SensorParams'   , 'NPMT PMTWL NSIPM SIPMWL'),
        ('ThresholdParams', 'thr_s1 thr_s2 thr_MAU thr_sipm thr_SIPM'),
        ('CalibratedSum'  , 'csum csum_mau'),
        ('PMaps'          , 'S1 S2 S2Si')):
    _add_namedtuple_in_this_module(name, attrs)

# Leave nothing but the namedtuple types in the namespace of this module
del name, namedtuple, sys, this_module, _add_namedtuple_in_this_module
