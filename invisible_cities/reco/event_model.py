# Clsses defining the event model

import numpy as np


class Cluster:
    """Represents a reconstructed cluster in the tracking plane"""
    def __init__(self, Q, xy, xy_rms, nsipm):

        self.Q        =  Q
        self._xy       =  xy
        self._xy_rms   =  xy_rms
        self.nsipm    =  nsipm

    @property
    def pos(self): return self._xy.pos

    @property
    def rms(self): return self._xy_rms.pos

    @property
    def X(self): return self._xy.x

    @property
    def Y(self): return self._xy.y

    @property
    def XY(self): return self._xy.XY

    @property
    def Xrms(self): return self._xy_rms.x

    @property
    def Yrms(self): return self._xy_rms.y

    @property
    def R(self): return self._xy.R

    @property
    def Phi(self): return self._xy.Phi

    def __str__(self):
        return """<nsipm = {} Q = {}
                    xy = {}  >""".format(self.__class__.__name__,
                                         self.nsipm, sel.Q, self._xy)
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
    def XYZ(self): return (self.X, self.Y, self.Z)

    @property
    def VXYZ(self): return np.array([self.X, self.Y, self.Z])

    @property
    def E(self): return self.s2_energy

    @property
    def Z(self): return self.z

    @property
    def npeak(self): return self.peak_number

    def __str__(self):
        return """<npeak = {} z = {} E = {} cluster ={} >""".format(self.__class__.__name__,
                    self.npeak, self.Z, sel.E, Cluster.__str())

    __repr__ =     __str__


class HitCollection:
    def __init__(self):


        self.hits = []

    def __str__(self):
        s =  "{}".format(self.__class__.__name__)
        s+= "Hit list:"
        s = [s + str(hit) for hit in self.hits]
        return s

    __repr__ =     __str__

class Event:
   def __init__(self):
       self.evt  = None
       self.time = None

   def __str__(self):
       s = "{0}Event\n{0}".format("#"*20 + "\n")
       for attr in self.__dict__:
           s += "{}: {}\n".format(attr, getattr(self, attr))
       return s

class KrEvent(Event):
    """Transient version of a point-like (Krypton) event."""
    def __init__(self):
        super().__init__()
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
