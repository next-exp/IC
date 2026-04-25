# Classes defining the event model

import tables as tb
import numpy  as np
import pandas as pd

from .. types.ic_types import NN
from .. types.ic_types import xy
from .. types.symbols  import HitEnergy
from .. core           import system_of_units as units
from .. core.core_functions import overflow_protection

from typing import List
from typing import Tuple
from typing import NamedTuple


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
    def Ehits(self): return self.hits[self.e_type].sum()

    @property
    def Etype(self): return self.e_type


class Cluster(BHit):
    """Represents a reconstructed cluster in the tracking plane"""
    def __init__(self, Q, xy, xy_var, nsipm, z, E=NN, Qc=-1):
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
        s2 = ['voxel number {} = {} \n'.format(i, voxel) for (i, voxel) in enumerate(self.voxels)]

        return  s + ''.join(s2)

    def __repr__(self):
        return self.__str__()


class Blob:
    """A Blob is a collection of Hits with a seed and a radius. """
    def __init__(self, seed: Tuple[float, float, float],
                       hits : pd.DataFrame,
                       radius : float,
                       e_type : HitEnergy = HitEnergy.E) ->None:
        self.seed   = seed
        self.hits   = hits
        self.E      = hits[e_type.value].sum()
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



kr_events_type = dict( event   = int
                     , time    = float
                     , s1_peak = np.uint16
                     , s2_peak = np.uint16
                     , nS1     = np.uint16
                     , nS2     = np.uint16
                     , S1w     = float
                     , S1h     = float
                     , S1e     = float
                     , S1t     = float
                     , S2w     = float
                     , S2h     = float
                     , S2e     = float
                     , S2q     = float
                     , S2t     = float
                     , qmax    = float
                     , Nsipm   = np.uint16
                     , DT      = float
                     , Z       = float
                     , X       = float
                     , Y       = float
                     , R       = float
                     , Phi     = float
                     , Xrms    = float
                     , Yrms    = float
                     )
