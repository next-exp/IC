# Clsses defining the event model

import numpy as np

from .. types.ic_types         import minmax
from .. core.exceptions        import PeakNotFound
from .. core.exceptions        import SipmEmptyList
from .. core.exceptions        import SipmNotFound
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


class Cluster:
    """Represents a reconstructed cluster in the tracking plane"""
    def __init__(self, Q, xy, xy_std, nsipm):
        self.Q       = Q
        self._xy     = xy
        self._xy_std = xy_std
        self.nsipm   = nsipm

    @property
    def pos (self): return self._xy.pos

    @property
    def std (self): return self._xy_std

    @property
    def X   (self): return self._xy.x

    @property
    def Y   (self): return self._xy.y

    @property
    def XY  (self): return self._xy.XY

    @property
    def Xrms(self): return np.sqrt(self._xy_std.x)

    @property
    def Yrms(self): return np.sqrt(self._xy_std.y)

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
                               cluster._xy, cluster._xy_std,
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

    def __str__(self):
        s =  "{}".format(self.__class__.__name__)
        s+= "Hit list:"
        s = [s + str(hit) for hit in self.hits]
        return s

    __repr__ =     __str__


# class PersistentHitCollection(HitCollection):
#     """Persistent version"""
#     def store(self, table):
#         row = table.row
#         for hit in self.hits:
#             row["event"] = self.event
#             row["time" ] = self.time
#             row["npeak"] = hit.npeak
#             row["nsipm"] = hit.nsipm
#             row["X"    ] = hit.X
#             row["Y"    ] = hit.Y
#             row["Xrms" ] = hit.Xrms
#             row["Yrms" ] = hit.Yrms
#             row["Z"    ] = hit.Z
#             row["Q"    ] = hit.Q
#             row["E"    ] = hit.E
#             row.append()


class KrEvent(Event):
    """Represents a point-like (Krypton) event."""
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

    def __str__(self):
        s = "{0}Event\n{0}".format("#"*20 + "\n")
        for attr in self.__dict__:
            s += "{}: {}\n".format(attr, getattr(self, attr))
        return s

    __repr__ =     __str__


# class PersistentKrEvent(KrEvent):
#     """Persistent version of KrEvent"""
#     def store(self, table):
#         row = table.row
#         for i in range(int(self.nS2)):
#             row["event"] = self.event
#             row["time" ] = self.time
#             row["peak" ] = i
#             row["nS2"  ] = self.nS2
#
#             row["S1w"  ] = self.S1w  [0]
#             row["S1h"  ] = self.S1h  [0]
#             row["S1e"  ] = self.S1e  [0]
#             row["S1t"  ] = self.S1t  [0]
#
#             row["S2w"  ] = self.S2w  [i]
#             row["S2h"  ] = self.S2h  [i]
#             row["S2e"  ] = self.S2e  [i]
#             row["S2q"  ] = self.S2q  [i]
#             row["S2t"  ] = self.S2t  [i]
#
#             row["Nsipm"] = self.Nsipm[i]
#             row["DT"   ] = self.DT   [i]
#             row["Z"    ] = self.Z    [i]
#             row["X"    ] = self.X    [i]
#             row["Y"    ] = self.Y    [i]
#             row["R"    ] = self.R    [i]
#             row["Phi"  ] = self.Phi  [i]
#             row["Xrms" ] = self.Xrms [i]
#             row["Yrms" ] = self.Yrms [i]
#             row.append()
