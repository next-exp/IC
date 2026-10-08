# Classes defining the event model

import numpy  as np

from .. types.ic_types import NN
from .. types.ic_types import xy


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


hit_type = dict( event = int
               , time  = float
               , npeak = np.uint16
               , Xpeak = float
               , Ypeak = float
               , X     = float
               , Y     = float
               , Z     = float
               , Q     = float
               , E     = float
               , Ec    = float
               )

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
