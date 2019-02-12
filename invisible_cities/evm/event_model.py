# Classes defining the event model

import tables as tb
import numpy  as np

from enum import Enum

from networkx                  import Graph
from .. types.ic_types         import NN
from .. types.ic_types         import minmax
from .. core.exceptions        import PeakNotFound
from .. core.exceptions        import SipmEmptyList
from .. core.exceptions        import SipmNotFound
from .. core.core_functions    import loc_elem_1d
from .. core.system_of_units_c import units

from typing import List
from typing import Tuple
from typing import NamedTuple

ZANODE = -9.425 * units.mm


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


class MCInfo(NamedTuple):
    """Transient class storing the tables of MC true info"""
    extents   : tb.Table
    hits      : tb.Table
    particles : tb.Table
    generators: tb.Table


class Waveform(NamedTuple):
    """Transient class storing times and charges for a sensor"""
    times     : np.ndarray
    charges   : np.ndarray
    bin_width : float


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


class MCParticle:
    """A MC Particle """

    def __init__(self, particle_name, primary,
                 mother_indx,
                 initial_vertex, final_vertex,
                 initial_volume, final_volume,
                 momentum, energy, creator_proc):
        self.name           = particle_name
        self.primary        = primary
        self.mother_indx    = mother_indx
        self.initial_vertex = initial_vertex
        self.final_vertex   = final_vertex
        self.initial_volume = initial_volume
        self.final_volume   = final_volume
        self.p              = momentum
        self.E              = energy
        self.process        = creator_proc
        self.hits           = []

    def __str__(self):
        return f""" MCParticle: name = {self.name}
                    initial volume = {self.initial_volume}, final volume = {self.final_volume}
                    kinetic energy = {self.E} MeV
                    mother = {self.mother_indx}, creator process = {self.process}
                    number of hits = {len(self.hits)}
                    \n"""

    __repr__ =  __str__


class HitEnergy(Enum):
    E        = 'E'
    Ec       = 'Ec'
    energy   = 'energy'
    energy_c = 'energy_c'


class BHit:
    """Base class representing a hit"""

    def __init__(self, x,y,z, E):
        self.xyz      = (x,y,z)
        self.energy   = E

    @property
    def XYZ  (self): return self.xyz

    @property
    def pos  (self): return np.array(self.xyz)

    @property
    def X   (self): return self.xyz[0]

    @property
    def Y   (self): return self.xyz[1]

    @property
    def Z   (self): return self.xyz[2]

    @property
    def E   (self): return self.energy

    def __str__(self):
        return '{}({.X}, {.Y}, {.Z}, E={.E})'.format(
            self.__class__.__name__, self, self, self, self)

    __repr__ =     __str__

    def __eq__(self, other):
        try:
            return np.array_equal(self.pos, other.pos) and self.E == other.E
        except AttributeError:
            return False

    def __hash__(self):
        return hash((self.E, tuple(self.pos)))


class MCHit(BHit):
    """Represents a MCHit"""
    def __init__(self, pos, t, E, l):
        super().__init__(pos[0],pos[1],pos[2], E)
        self.time          = t
        self.label         = l

    @property
    def T     (self): return self.time

    @property
    def Label (self): return self.label

    def __str__(self):
        return '<label = {}, pos = {}, E = {}, time = {}>'.format(self.label,
                self.pos.tolist(), self.E, self.time)

    __repr__ =     __str__

    def __eq__(self, other):
        try:
            return np.array_equal(self.pos, other.pos) and self.E == other.E and self.time == other.time
        except AttributeError:
            return False

    def __hash__(self):
        return hash((self.E, self.time, self.xyz))


class Voxel(BHit):
    """Represents a Voxel"""
    def __init__(self, x,y,z, E, size, hits=[], e_type : HitEnergy = HitEnergy.E.value):
        super().__init__(x,y,z, E)
        self._size  = size
        self.hits   = hits
        self.e_type = e_type

    @property
    def size(self): return self._size

    @property
    def Ehits(self): return sum(getattr(h, self.e_type) for h in self.hits)

    @property
    def Etype(self): return self.e_type


class Cluster(BHit):
    """Represents a reconstructed cluster in the tracking plane"""
    def __init__(self, Q, xy, xy_var, nsipm, z=ZANODE, E=NN, Ql=-1, Qc=-1):
        if E == NN:
            super().__init__(xy.x, xy.y, z, Q)
        else:
            super().__init__(xy.x, xy.y, z, E)

        self.Q       = Q
        self.Ql      = Ql
        self.Qc      = Qc
        self._xy     = xy
        self._xy_var = xy_var
        self.nsipm   = nsipm

    @property
    def posxy (self): return self._xy.pos

    @property
    def var (self): return self._xy_var

    @property
    def XY  (self): return self._xy.XY

    @property
    def Xrms(self): return np.sqrt(self._xy_var.x)

    @property
    def Yrms(self): return np.sqrt(self._xy_var.y)

    @property
    def R   (self): return self._xy.R

    @property
    def Phi (self): return self._xy.Phi

    def __str__(self):
        return """< nsipm = {} Q = {}
                    xy = {} 3dHit = {}  >""".format(self.nsipm, self.Q, self._xy,
                                                     super().__str__())
    __repr__ =     __str__


class Hit(Cluster):
    """Represents a reconstructed hit (cluster + z + energy)"""
    def __init__(self, peak_number, cluster, z, s2_energy, peak_xy,
                s2_energy_l=-1, s2_energy_c=-1, zc=ZANODE):


        super().__init__(cluster.Q,
                         cluster._xy, cluster._xy_var,
                         cluster.nsipm, z, s2_energy, cluster.Ql, cluster.Qc)

        self.peak_number = peak_number
        self.Xpeak = peak_xy.x
        self.Ypeak = peak_xy.y
        self.energy_l = s2_energy_l
        self.energy_c = s2_energy_c
        self.z_c      = zc

    @property
    def npeak(self): return self.peak_number

    @property
    def El(self): return self.energy_l

    @property
    def Ec(self): return self.energy_c

    @property
    def Zc(self): return self.z_c

    def __str__(self):
        return """<{} : npeak = {} z = {} XYpeak = {}, {} E = {} cluster ={} >""".format(self.__class__.__name__,
                    self.npeak, self.Z, self.Xpeak, self.Ypeak, self.E, super().__str__())

    __repr__ =     __str__


class VoxelCollection:
    """A collection of voxels. """
    def __init__(self, voxels : List[Voxel]):
        self.voxels = voxels
        self.E = sum(v.E for v in voxels)

    @property
    def number_of_voxels(self):
        return len(self.voxels)

    def __str__(self):
        s =  "VoxelCollection: (number of voxels = {})\n".format(self.number_of_voxels)
        s2 = ['voxel number {} = {} \n'.format((i, voxel) for (i, voxel) in enumerate(self.voxels))]

        return  s + ''.join(s2)

    def __repr__(self):
        return self.__str__()


class Blob():
    """A Blob is a collection of Hits with a seed and a radius. """
    def __init__(self, seed: Tuple[float, float, float],
                       hits : List[BHit],
                       radius : float,
                       e_type : HitEnergy = HitEnergy.E.value) ->None:
        super().__init__(hits)
        self.seed   = seed
        self.hits   = hits
        self.energy = sum(getattr(h, e_type) for h in hits)
        self.radius = radius
        self.e_type = e_type

    @property
    def Etype(self): return self.e_type

    def __str__(self):
        s =  """Blob: (hits = {} \n
                seed   = {} \n
                blob energy = {} \n
                blob radius = {}
        """.format(self.hits, self.seed, self.energy, self.radius)

        return  s

    def __repr__(self):
        return self.__str__()


class Track(VoxelCollection):
    """A track is a collection of linked voxels. """
    def __init__(self, voxels : List[Voxel], blobs: Tuple[Blob, Blob]) ->None:
        super().__init__(voxels)
        self.blobs = blobs

    def __str__(self):
        s =  """Track: (number of voxels = {})\n,
                voxels = {} \n
                blob_a = {} \n
                blob_b = {}
        """.format(self.number_of_voxels, self.voxels, self.blobs[0], self.blobs[1])

        return  s

    def __repr__(self):
        return self.__str__()


class TrackCollection(Event):
    """A Collection of tracks"""
    def __init__(self, event_number, event_time):
        Event.__init__(self, event_number, event_time)
        self.tracks = []

    @property
    def number_of_tracks(self):
        return len(self.tracks)

    def store(self, table):
        row = table.row
        for i, t in enumerate(self.tracks):
            row["event"]    = self.event
            row["time" ]    = self.time
            row["track_no"] = i

            for j, voxel in enumerate(t.voxels):
                row["voxel_no"] = j
                row["X"    ] = voxel.X
                row["Y"    ] = voxel.Y
                row["Z"    ] = voxel.Z
                row["E"    ] = voxel.E

                row.append()

    def __str__(self):
        s =  "{}".format(self.__class__.__name__)
        s+= "Track list:"
        s2 = [str(trk) for trk in self.tracks]
        return  s + ''.join(s2)

    __repr__ =     __str__


class HitCollection(Event):
    """A Collection of hits"""
    def __init__(self, event_number, event_time):
        Event.__init__(self, event_number, event_time)
        self.hits = []

    def store(self, table):
        row = table.row
        for hit in self.hits:
            row["event"] = self.event
            row["time" ] = self.time
            row["npeak"] = hit.npeak
            row["Xpeak"] = hit.Xpeak
            row["Ypeak"] = hit.Ypeak
            row["nsipm"] = hit.nsipm
            row["X"    ] = hit.X
            row["Y"    ] = hit.Y
            row["Xrms" ] = hit.Xrms
            row["Yrms" ] = hit.Yrms
            row["Z"    ] = hit.Z
            row["Q"    ] = hit.Q
            row["E"    ] = hit.E
            row["Ql"   ] = hit.Ql
            row["El"   ] = hit.El
            row["Qc"   ] = hit.Qc
            row["Ec"   ] = hit.Ec
            row["Zc"   ] = hit.Zc
            row.append()

    def __str__(self):
        s =  "{}".format(self.__class__.__name__)
        s+= "Hit list:"
        s2 = [str(hit) for hit in self.hits]
        return  s + ''.join(s2)

    __repr__ =     __str__


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
        self.DT    = [] # drift time
        self.Z     = [] # Position (x,y,z,R,phi)
        self.X     = []
        self.Y     = []
        self.R     = []
        self.Phi   = []
        self.Xrms  = [] # error in position
        self.Yrms  = []
        self.Zrms  = []

    def fill_defaults(self):
        if self.nS1 == 0:
            for attribute in ["w", "h", "e", "t"]:
                setattr(self, "S1" + attribute, [np.nan])

        if self.nS2 == 0:
            for attribute in ["w", "h", "e", "t", "q"]:
                setattr(self, "S2" + attribute, [np.nan])

            self.Nsipm = 0
            for attribute in ["X", "Y", "R", "Phi", "Xrms", "Yrms", "Zrms"]:
                setattr(self, attribute, [np.nan])

        if not self.nS1 or not self.nS2:
            for attribute in ["Z", "DT"]:
                setattr(self, attribute, [[np.nan] * max(self.nS1, 1)] * max(self.nS2, 1))

    def store(self, table):
        row = table.row

        s1_peaks = range(int(self.nS1)) if self.nS1 else [-1]
        s2_peaks = range(int(self.nS2)) if self.nS2 else [-1]
        self.fill_defaults()

        for i in s1_peaks:
            for j in s2_peaks:
                row["event"  ] = self.event
                row["time"   ] = self.time
                row["s1_peak"] = i
                row["s2_peak"] = j
                row["nS1"    ] = self.nS1
                row["nS2"    ] = self.nS2

                row["S1w"    ] = self.S1w  [i]
                row["S1h"    ] = self.S1h  [i]
                row["S1e"    ] = self.S1e  [i]
                row["S1t"    ] = self.S1t  [i]

                row["S2w"    ] = self.S2w  [j]
                row["S2h"    ] = self.S2h  [j]
                row["S2e"    ] = self.S2e  [j]
                row["S2q"    ] = self.S2q  [j]
                row["S2t"    ] = self.S2t  [j]

                row["Nsipm"  ] = self.Nsipm[j]
                row["DT"     ] = self.DT   [j][i]
                row["Z"      ] = self.Z    [j][i]
                row["Zrms"   ] = self.Zrms [j]
                row["X"      ] = self.X    [j]
                row["Y"      ] = self.Y    [j]
                row["R"      ] = self.R    [j]
                row["Phi"    ] = self.Phi  [j]
                row["Xrms"   ] = self.Xrms [j]
                row["Yrms"   ] = self.Yrms [j]
                row.append()

    def __str__(self):
        s = "{0}Event\n{0}".format("#"*20 + "\n")
        for attr in self.__dict__:
            s += "{}: {}\n".format(attr, getattr(self, attr))
        return s

    __repr__ =     __str__
