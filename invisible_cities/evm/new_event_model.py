import numpy as np

from pytest import approx


class xy:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def r(self):
        return (self.x**2 + self.y**2)**0.5

    @property
    def phi(self):
        return np.arctan2(self.y, self.x)

    @property
    def xy(self):
        return self.x, self.y


class xyz(xy):
    def __init__(self, x, y, z):
        super().__init__(x, y)
        self.z = z

    @property
    def xyz(self):
        return self.x, self.y, self.z


class xyzE(xyz):
    def __init__(self, x, y, z, E):
        super().__init__(x, y, z)
        self.E = E

    @property
    def xyze(self):
        return self.x, self.y, self.z, self.E


class MCHit(xyzE):
    def __init__(self, x, y, z, E, t):
        super().__init__(x, y, z, E)
        self.t = t

    @property
    def xyzt(self):
        return self.x, self.y, self.z, self.t

    @property
    def xyzet(self):
        return self.x, self.y, self.z, self.E, self.t


class SpaceElement(xyzE):
    def __init__(self, x, y, z, E, size, hit_es):
        if not np.sum(hit_es) == approx(E):
            classname = self.__class__.__name__
            raise ValueError(f"Hit energies must add up to {classname} energy")

        super().__init__(x, y, z, E)
        self.size   = size
        self.hit_es = np.array(hit_es)


class            Voxel(SpaceElement): pass
class       CubicVoxel(SpaceElement): pass
class EllipsoidalVoxel(SpaceElement): pass
class   SphericalVoxel(SpaceElement): pass


class Cluster(xy):
    def __init__(self, Q, x, y, x_var, y_var, n_sipm):
        super().__init__(x, y)
        self.Q      = Q
        self.x_var  = x_var
        self.y_var  = y_var
        self.n_sipm = n_sipm

    @property
    def x_rms(self):
        return self.x_var**0.5

    @property
    def y_rms(self):
        return self.y_var**0.5

    @property
    def xy_var(self):
        return self.x_var, self.y_var

    @property
    def xy_rms(self):
        return self.x_rms, self.y_rms


class Hit(xyzE):
    def __init__(self, cluster, z, E, peak_no):
        xyzE.__init__(self, cluster.x, cluster.y, z, E)
        self.cluster = cluster
        self.peak_no = peak_no


class VoxelCollection:
    def __init__(self, voxels):
        self.voxels = voxels
        self.E      = np.sum(voxel.E for voxel in self.voxels)


class Blob(VoxelCollection):
    def __init__(self, voxels, seed, size):
        super().__init__(voxels)
        self.seed = seed
        self.size = size


class SphericalBlob(Blob): pass
class     CubicBlob(Blob): pass


class Track(VoxelCollection):
    def __init__(self, voxels, blobs, extrema):
        super().__init__(voxels)
        self.blobs   = blobs
        self.extrema = extrema

    @property
    def smallest_blob(self):
        index = np.argmin([blob.E for blob in self.blobs])
        return self.blobs[index]


    @property
    def  biggest_blob(self):
        index = np.argmax([blob.E for blob in self.blobs])
        return self.blobs[index]


class Event:
    def __init__(self, event_number, timestamp):
        self.event_number = event_number
        self.timestamp    = timestamp


class HitCollection(Event):
    def __init__(self, event_number, timestamp, hits):
        super().__init__(event_number, timestamp)
        self.hits = hits


class TrackCollection(Event):
    def __init__(self, event_number, timestamp, tracks):
        super().__init__(event_number, timestamp)
        self.tracks = tracks


class PointlikeEvent(Event):
    def __init__(self, event_number, timestamp):
        super().__init__(event_number, timestamp)

        self.nS1   = -1 # number of S1s in the event
        self.S1w   = [] # S1 widhts
        self.S1h   = [] # S1 heigths
        self.S1e   = [] # S1 energys
        self.S1t   = [] # S1 times

        self.nS2   = -1 # number of S2s in the event
        self.S2w   = [] # S2 widhts
        self.S2h   = [] # S2 heigths
        self.S2e   = [] # S2 energys
        self.S2q   = [] # S2 charges
        self.S2t   = [] # S2 times

        self.Nsipm = [] # number of SiPMs for each S2 peak
        self.DT    = [] # drift times (wrt the first S1)
        self.Z     = [] # positions in each coordinate
        self.X     = [] # |
        self.Y     = [] # |
        self.R     = [] # |
        self.Phi   = [] # |
        self.Xrms  = [] # std devs for each reconstructed coordinate
        self.Yrms  = [] # |
        self.Zrms  = [] # |
