# Clsses defining the event model

from functools import reduce

import numpy as np

from .. core.ic_types          import minmax
from .. core.core_functions    import loc_elem_1d
from .. core.system_of_units_c import units

class SensorParams:
    """Transient class storing sensor parameters."""
    def __init__(self, npmt, pmtwl, nsipm, sipmwl):
       self.npmt   = npmt
       self.pmtwl  = pmtwl
       self.nsipm  = nsipm
       self.sipmwl = sipmwl

    @property
    def NPMT  (self): return self.npmt

    @property
    def PMTWL (self): return self.pmtwl

    @property
    def NSIPM (self): return self.nsipm

    @property
    def SIPMWL(self): return self.sipmwl

    def __str__(self):
       s = "{0}SensorParams\n{0}".format("#"*20 + "\n")
       for attr in self.__dict__:
           s += "{}: {}\n".format(attr, getattr(self, attr))
       return s

    __repr__ = __str__


class Event:
    """Transient class storing event and time info."""
    def __init__(self, event_number, event_time):
       self.event = event_number
       self.time  = event_time

    def __str__(self):
       s = "{0}Event\n{0}".format("#"*20 + "\n")
       for attr in self.__dict__:
           s += "{}: {}\n".format(attr, getattr(self, attr))
       return s

    __repr__ = __str__


class Waveform:
    """Transient class representing a waveform.

    A Waveform is represented as a pair of arrays:
    t: np.array() describing time bins
    E: np.array() describing energy.
    """
    def __init__(self, t, E):
        assert len(t) == len(E)
        self.t            = np.array(t)
        self.E            = np.array(E)
        self.total_energy = np.sum(self.E)
        self.height       = np.max(self.E)
        self._tm          = minmax(self.t[0], self.t[-1])
        self._i_t         = loc_elem_1d(self.E, self.height)

    @property
    def number_of_samples(self): return len(self.t)

    @property
    def tpeak(self): return self.t(self._i_t)

    @property
    def tmin_tmax(self): return self._tm

    @property
    def width(self): return self._tm.bracket

    def __str__(self):
        s = """Waveform(samples = {} width = {} ns , energy = {} pes
        height = {} pes tmin-tmax = {} mus """.format(self.number_of_samples,
        self.width, self.total_energy, self.height,
        self.tmin_tmax * (1/ units.mus))
        return s

    __repr__ = __str__


class S12:
    """Transient class representing an S1/S2 signal
    The S12 attribute is a dictionary s12
    {i: Waveform(t,E)}, where i is peak number.

    """

    def __init__(self, s12, s12_type='S1'):
        self._s12 = {i: Waveform(t, E) for i, (t,E) in s12.items()}
        self.type = s12_type

    @property
    def number_of_peaks(self): return len(self._s12)

    @property
    def peaks(self): return self._s12

    def peak_waveform(self, i):
        return self._s12[i]

    def __str__(self):
        s =  "S12(type = {}, number of peaks = {})\n".format(self.type,
                                                      self.number_of_peaks)

        s2 = ['peak number = {}: {} \n'.format(i,
                                    self.peak_waveform(i)) for i in self.peaks]

        return reduce(lambda s, x: s + x, s2, s)

    __repr__ = __str__


class Cluster:
    """Represents a reconstructed cluster in the tracking plane"""
    def __init__(self, Q, xy, xy_rms, nsipm):
        self.Q       = Q
        self._xy     = xy
        self._xy_rms = xy_rms
        self.nsipm   = nsipm

    @property
    def pos (self): return self._xy.pos

    @property
    def rms (self): return self._xy_rms.pos

    @property
    def X   (self): return self._xy.x

    @property
    def Y   (self): return self._xy.y

    @property
    def XY  (self): return self._xy.XY

    @property
    def Xrms(self): return self._xy_rms.x

    @property
    def Yrms(self): return self._xy_rms.y

    @property
    def R   (self): return self._xy.R

    @property
    def Phi (self): return self._xy.Phi

    def __str__(self):
        return """<nsipm = {} Q = {}
                    xy = {}  >""".format(self.__class__.__name__,
                                         self.nsipm, self.Q, self._xy)
    __repr__ =     __str__


class Hit(Cluster):
    """Represents a reconstructed hit (cluster + z + energy)"""
    def __init__(self, peak_number, cluster, z, s2_energy):

        Cluster.__init__(self, cluster.Q,
                               cluster._xy, cluster._xy_rms,
                               cluster.nsipm)

        self.peak_number = peak_number
        self.z           = z
        self.s2_energy   = s2_energy

    @property
    def XYZ  (self): return (self.X, self.Y, self.Z)

    @property
    def VXYZ (self): return np.array([self.X, self.Y, self.Z])

    @property
    def E    (self): return self.s2_energy

    @property
    def Z    (self): return self.z

    @property
    def npeak(self): return self.peak_number

    def __str__(self):
        return """<npeak = {} z = {} E = {} cluster ={} >""".format(self.__class__.__name__,
                    self.npeak, self.Z, self.E, Cluster.__str())

    __repr__ =     __str__


class HitCollection(Event):
    """Transient version of Hit Collection"""
    def __init__(self, event_number, event_time):
        Event.__init__(self, event_number, event_time)
        self.hits = []

    def __str__(self):
        s =  "{}".format(self.__class__.__name__)
        s+= "Hit list:"
        s = [s + str(hit) for hit in self.hits]
        return s

    __repr__ =     __str__


class PersistentHitCollection(HitCollection):
    """Persistent version"""
    def store(self, table):
        row = table.row
        for hit in self.hits:
            row["event"] = self.event
            row["time" ] = self.time
            row["npeak"] = hit.npeak
            row["nsipm"] = hit.nsipm
            row["X"    ] = hit.X
            row["Y"    ] = hit.Y
            row["Xrms" ] = hit.Xrms
            row["Yrms" ] = hit.Yrms
            row["Z"    ] = hit.Z
            row["Q"    ] = hit.Q
            row["E"    ] = hit.E
            row.append()

class KrEvent(Event):
    """Transient version of a point-like (Krypton) event."""
    def __init__(self, event_number, event_time):
        Event.__init__(self, event_number, event_time)
        self.nS1   = -1 # number of S1 in the event
        self.S1w   = [] # widht
        self.S1h   = [] # heigth
        self.S1e   = [] # energy
        self.S1t   = [] # time

        self.nS2   = -1 # number of S2s in the event
        self.S2w   = []
        self.S2h   = []
        self.S2e   = []
        self.S2q   = [] # Charge in the S2Si
        self.S2t   = [] # time

        self.Nsipm = [] # number of SiPMs in S2Si
        self.DT    = [] # drift time (wrt S1[0])
        self.Z     = [] # Position (x,y,z,R,phi)
        self.X     = []
        self.Y     = []
        self.R     = []
        self.Phi   = []
        self.Xrms  = [] # error in position
        self.Yrms  = []

    def __str__(self):
        s = "{0}Event\n{0}".format("#"*20 + "\n")
        for attr in self.__dict__:
            s += "{}: {}\n".format(attr, getattr(self, attr))
        return s

class PersistentKrEvent(KrEvent):
    """Persistent version of KrEvent"""
    def store(self, table):
        row = table.row
        for i in range(int(self.nS2)):
            row["event"] = self.event
            row["time" ] = self.time
            row["peak" ] = i
            row["nS2"  ] = self.nS2

            row["S1w"  ] = self.S1w  [0]
            row["S1h"  ] = self.S1h  [0]
            row["S1e"  ] = self.S1e  [0]
            row["S1t"  ] = self.S1t  [0]

            row["S2w"  ] = self.S2w  [i]
            row["S2h"  ] = self.S2h  [i]
            row["S2e"  ] = self.S2e  [i]
            row["S2q"  ] = self.S2q  [i]
            row["S2t"  ] = self.S2t  [i]

            row["Nsipm"] = self.Nsipm[i]
            row["DT"   ] = self.DT   [i]
            row["Z"    ] = self.Z    [i]
            row["X"    ] = self.X    [i]
            row["Y"    ] = self.Y    [i]
            row["R"    ] = self.R    [i]
            row["Phi"  ] = self.Phi  [i]
            row["Xrms" ] = self.Xrms [i]
            row["Yrms" ] = self.Yrms [i]
            row.append()
