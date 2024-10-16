from enum import auto
from enum import Enum
from enum import EnumMeta

from . ic_types import AutoNameEnumBase


class BlsMode(AutoNameEnumBase):
    mean      = auto()
    median    = auto()
    scipymode = auto()
    mode      = auto()


class Contiguity(Enum):
    FACE   = 1.2
    EDGE   = 1.5
    CORNER = 1.8


class CutType(AutoNameEnumBase):
    abs = auto()
    rel = auto()


class DarkModel(AutoNameEnumBase):
    mean      = auto()
    threshold = auto()


class DeconvolutionMode(AutoNameEnumBase):
    joint    = auto()
    separate = auto()


class EventRange(AutoNameEnumBase):
    all  = auto()
    last = auto()

all_events = EventRange.all # particularly helpful


class HitEnergy(AutoNameEnumBase):
    E  = auto()
    Ec = auto()
    Ep = auto()


class InterpolationMethod(AutoNameEnumBase):
    nearest         = auto()
    linear          = auto()
    cubic           = auto()
    nointerpolation = auto()


class KrFitFunction(AutoNameEnumBase):
    expo    = auto()
    linear  = auto()
    log_lin = auto()


class MCTableType(AutoNameEnumBase):
    configuration    = auto()
    events           = auto()
    event_mapping    = auto()
    extents          = auto()
    generators       = auto()
    hits             = auto()
    particles        = auto()
    sensor_positions = auto()
    sns_positions    = auto()
    sns_response     = auto()
    waveforms        = auto()
    string_map       = auto()


class NormStrategy(AutoNameEnumBase):
    mean   = auto()
    max    = auto()
    kr     = auto()
    custom = auto()


class NormMode(AutoNameEnumBase):
    first  = auto()
    second = auto()
    sumof  = auto()
    mean   = auto()


class PMTCalibMode(AutoNameEnumBase):
    gain          = auto()
    gain_maw      = auto()
    gain_nodeconv = auto()


class RebinMethod(AutoNameEnumBase):
    stride    = auto()
    threshold = auto()


class SensorType(AutoNameEnumBase):
    SIPM = auto()
    PMT  = auto()


class SiPMCalibMode(AutoNameEnumBase):
    subtract_mode             = auto()
    subtract_median           = auto()
    subtract_mode_calibrate   = auto()
    subtract_mean_calibrate   = auto()
    subtract_median_calibrate = auto()
    subtract_mode_zs          = auto()


class SiPMCharge(AutoNameEnumBase):
    raw             = auto()
    signal_to_noise = auto()



class SiPMThreshold(AutoNameEnumBase):
    common     = auto()
    individual = auto()


class XYReco(AutoNameEnumBase):
    barycenter = auto()
    corona     = auto()


class WfType(AutoNameEnumBase):
    rwf  = auto()
    mcrd = auto()


ALL_SYMBOLS = {}
for etype in dict(locals()).values():
    if not isinstance(etype, EnumMeta): continue
    if etype is EnumMeta              : continue
    if etype is Enum                  : continue
    if etype is AutoNameEnumBase      : continue

    for symbol in etype:
        ALL_SYMBOLS[symbol.name] = symbol
