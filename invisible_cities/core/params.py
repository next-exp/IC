from collections import namedtuple

S12Params       = namedtuple('S12Params', 'tmin tmax stride lmin lmax rebin')
SensorParams    = namedtuple('SensorParam', 'NPMT PMTWL NSIPM SIPMWL')
ThresholdParams = namedtuple('ThresholdParams',
                             'thr_s1 thr_s2 thr_MAU thr_sipm thr_SIPM')
CalibratedSum   = namedtuple('CalibratedSum', 'csum csum_mau')
PMaps           = namedtuple('PMaps', 'S1 S2 S2Si')
