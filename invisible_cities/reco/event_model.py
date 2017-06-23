# Clsses defining the event model

import numpy as np

from .. core.ic_types          import minmax
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


class Waveform:
    """Transient class representing a waveform.

    A Waveform is represented as a pair of arrays:
    t: np.array() describing time bins
    E: np.array() describing energy.
    """
    def __init__(self, t, E):
        assert len(t) == len(E)
        self.t            = (np.array(t) if not np.any(np.isnan(t))
                             else np.array([0]))
        self.E            = (np.array(E) if not np.any(np.isnan(E))
                             else np.array([0]))
        self.total_energy = np.sum(self.E)
        self.height       = np.max(self.E)
        self._tm          = minmax(self.t[0], self.t[-1])

        self._i_t         = (loc_elem_1d(self.E, self.height)
                             if self.total_energy > 0
                             else 0)
    @property
    def good_waveform(self): return (False
                                    if np.array_equal(self.t, np.array([0])) or
                                       np.array_equal(self.E, np.array([0]))
                                    else True )

    @property
    def number_of_samples(self): return len(self.t)

    @property
    def tpeak(self): return self.t[self._i_t]

    @property
    def tmin_tmax(self): return self._tm

    @property
    def width(self): return self._tm.bracket

    def __eq__(self, other):
        return np.all(self.t == other.t) and np.all(self.E == other.E)

    def __str__(self):
        s = """Waveform(samples = {} width = {:.1f} mus , energy = {:.1f} pes
        height = {:.1f} pes tmin-tmax = {} mus """.format(self.number_of_samples,
        self.width / units.mus, self.total_energy, self.height,
        (self.tmin_tmax * (1 / units.mus)).__str__(1))
        return s

    __repr__ = __str__


class _S12:
    """Base class representing an S1/S2 signal
    The S12 attribute is a dictionary s12
    {i: Waveform(t,E)}, where i is peak number.

    The notation _S12 is intended to make this
    class private (public classes S1 and S2 will
    extend it).

    The rationale to use S1 and S2 rather than a single
    class S12 to represent both S1 and S2 is that, although
    structurally identical, S1 and S2 represent quite
    different objects. In Particular an S2Si is constructed
    with a S2 not a S1.

    An S12 is represented as a dictinary of Peaks, where each
    peak is a waveform.

    """

    def __init__(self, s12d):
        """Takes an s12d ={peak_number:[[t], [E]]}"""
        self.peaks = {i: Waveform(t, E) for i, (t,E) in s12d.items()}

    @property
    def number_of_peaks(self): return len(self.peaks)

    def peak_waveform(self, peak_number):
        try:
            return self.peaks[peak_number]
        except KeyError:
             raise PeakNotFound

    def __str__(self):
        s =  "{}(number of peaks = {})\n".format(self.__class__.__name__, self.number_of_peaks)

        s2 = ['peak number = {}: {} \n'.format(i,
                                    self.peak_waveform(i)) for i in self.peaks]

        return  s + ''.join(s2)

    __repr__ = __str__


# These types merely serve to distinguish the different meanings of
# isomorphic data structures.
class S1(_S12):
    """Transient class representing an S1 signal."""


class S2(_S12):
    """Transient class representing an S2 signal."""


class S2Si(S2):
    """Transient class representing the combination of
    S2 and the SiPM information.
    Notice that S2Si is constructed using an s2d and an s2sid.
    The s2d is an s12 dictionary (not an S2 instance)
    The s2sid is a dictionary {peak:{nsipm:[E]}}
    """

    def __init__(self, s2d, s2sid):
        """where:
           s2d   = {peak_number:[[t], [E]]}
           s2sid = {peak:{nsipm:[Q]}}
           Q is the energy in each SiPM sample
        """
        S2.__init__(self, s2d)
        self._s2sid = s2sid

    def number_of_sipms_in_peak(self, peak_number):
        return len(self._s2sid[peak_number])

    def sipms_in_peak(self, peak_number):
        try:
            return tuple(self._s2sid[peak_number].keys())
        except KeyError:
            raise PeakNotFound

    def sipm_waveform(self, peak_number, sipm_number):
        if self.number_of_sipms_in_peak(peak_number) == 0:
            raise SipmEmptyList
        try:
            E = self._s2sid[peak_number][sipm_number]
            return Waveform(self.peak_waveform(peak_number).t, E)
        except KeyError:
            raise SipmNotFound

    def sipm_waveform_zs(self, peak_number, sipm_number):
        if self.number_of_sipms_in_peak(peak_number) == 0:
            raise SipmEmptyList("No SiPMs associated to this peak")
        try:
            E = self._s2sid[peak_number][sipm_number]
            t = self.peak_waveform(peak_number).t[E>0]
            return Waveform(t, E[E>0])
        except KeyError:
            raise SipmNotFound

    def sipm_total_energy(self, peak_number, sipm_number):
        if self.number_of_sipms_in_peak(peak_number) == 0:
            raise SipmEmptyList("No SiPMs associated to this peak")
        try:
            et = np.sum(self._s2sid[peak_number][sipm_number])
            return et
        except KeyError:
            raise SipmNotFound

    def __str__(self):
        s  = "=" * 80 + "\n" + S2.__str__(self)

        s += "-" * 80 + "\nSiPMs for non-empty peaks\n\n"

        s2a = ["peak number = {}: nsipm in peak = {}"
               .format(peak_number, self.sipms_in_peak(peak_number))
               for peak_number in self.peaks
               if len(self.sipms_in_peak(peak_number)) > 0]

        s += '\n\n'.join(s2a) + "\n"

        s += "-" * 80 + "\nSiPMs Waveforms\n\n"

        s2b = ["peak number = {}: sipm number = {}\n    sipm waveform (zs) = {}".format(peak_number, sipm_number, self.sipm_waveform_zs(peak_number, sipm_number))
               for peak_number in self.peaks
               for sipm_number in self.sipms_in_peak(peak_number)
               if len(self.sipms_in_peak(peak_number)) > 0]

        return s + '\n'.join(s2b) + "\n" + "=" * 80

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
