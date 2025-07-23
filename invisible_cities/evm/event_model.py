# Classes defining the event model

import tables as tb
import numpy  as np

from .. types.ic_types import NN
from .. types.ic_types import xy
from .. types.symbols  import HitEnergy
from .. core           import system_of_units as units

from typing import List
from typing import Tuple
from typing import NamedTuple

ZANODE = -9.425 * units.mm


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


class BHit:
    """Base class representing a hit"""

    def __init__(self, x,y,z, E):
        self.xyz      = (x,y,z)
        self.E        = E

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

    def __str__(self):
        return '{}({.X}, {.Y}, {.Z}, E={.E})'.format(
            self.__class__.__name__, self, self, self, self)

    __repr__ =     __str__


class MCHit(BHit):
    """Represents a MCHit"""
    def __init__(self, pos, t, E, l):
        super().__init__(pos[0],pos[1],pos[2], E)
        self.time          = t
        self.label         = l


    def __str__(self):
        return '<label = {}, pos = {}, E = {}, time = {}>'.format(self.label,
                self.pos.tolist(), self.E, self.time)

    __repr__ =     __str__


class Voxel(BHit):
    """Represents a Voxel"""
    def __init__(self, x,y,z, E, size, hits=None, e_type : HitEnergy = HitEnergy.E):
        super().__init__(x,y,z, E)
        self._size  = size
        self.hits   = hits if hits is not None else []
        self.e_type = e_type.value

    @property
    def size(self): return self._size

    @property
    def Ehits(self): return sum(getattr(h, self.e_type) for h in self.hits)

    @property
    def Etype(self): return self.e_type


class Cluster(BHit):
    """Represents a reconstructed cluster in the tracking plane"""
    def __init__(self, Q, xy, xy_var, nsipm, z=ZANODE, E=NN, Qc=-1):
        if E == NN:
            super().__init__(xy.x, xy.y, z, Q)
        else:
            super().__init__(xy.x, xy.y, z, E)

        self.Q       = Q
        self.Qc      = Qc
        self._xy     = xy
        self._xy_var = xy_var
        self.nsipm   = nsipm

    def empty():
        return Cluster(NN, xy.empty(), xy.zero(), 0)

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
                 s2_energy_c=-1, track_id=-1, Ep=-1):


        super().__init__(cluster.Q,
                         cluster._xy, cluster._xy_var,
                         cluster.nsipm, z, s2_energy, cluster.Qc)

        self.peak_number = peak_number
        self.Xpeak       = peak_xy.x
        self.Ypeak       = peak_xy.y
        self.Ec          = s2_energy_c
        self.track_id    = track_id
        self.Ep          = Ep

    @property
    def npeak(self): return self.peak_number

    def __str__(self):
        return """<{} : npeak = {} z = {} XYpeak = {}, {} E = {} Ec = {} Ep = {} trackid = {} cluster ={} >""".format(self.__class__.__name__,
                    self.npeak, self.Z, self.Xpeak, self.Ypeak, self.E, self.Ec, self.Ep, self.track_id, super().__str__())

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


class Blob:
    """A Blob is a collection of Hits with a seed and a radius. """
    def __init__(self, seed: Tuple[float, float, float],
                       hits : List[BHit],
                       radius : float,
                       e_type : HitEnergy = HitEnergy.E) ->None:
        self.seed   = seed
        self.hits   = hits
        self.E      = sum(getattr(h, e_type.value) for h in hits)
        self.radius = radius
        self.e_type = e_type.value

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


class HitCollection(Event):
    """A Collection of hits"""
    def __init__(self, event_number, event_time, hits=None):
        Event.__init__(self, event_number, event_time)
        self.hits = [] if hits is None else hits

    def store(self, table):
        row = table.row
        for hit in self.hits:
            row["event"   ] = self.event
            row["time"    ] = self.time
            row["npeak"   ] = hit .npeak
            row["Xpeak"   ] = hit .Xpeak
            row["Ypeak"   ] = hit .Ypeak
            row["nsipm"   ] = hit .nsipm
            row["X"       ] = hit .X
            row["Y"       ] = hit .Y
            row["Xrms"    ] = hit .Xrms
            row["Yrms"    ] = hit .Yrms
            row["Z"       ] = hit .Z
            row["Q"       ] = hit .Q
            row["E"       ] = hit .E
            row["Qc"      ] = hit .Qc
            row["Ec"      ] = hit .Ec
            row["track_id"] = hit .track_id
            row["Ep"      ] = hit .Ep
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
        self.qmax  = []

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

            self.Nsipm = [0]
            self.qmax  = [0]
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
                row["qmax"   ] = self.qmax [j]

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
