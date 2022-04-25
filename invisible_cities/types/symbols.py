from enum import auto
from enum import Enum

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
    nearest = auto()
    linear  = auto()
    cubic   = auto()
    none    = auto()


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


class RebinMethod(AutoNameEnumBase):
    stride    = auto()
    threshold = auto()


class SensorType(AutoNameEnumBase):
    SIPM = auto()
    PMT  = auto()


class SiPMCharge(AutoNameEnumBase):
    raw             = auto()
    signal_to_noise = auto()


class WfType(AutoNameEnumBase):
    rwf  = auto()
    mcrd = auto()
