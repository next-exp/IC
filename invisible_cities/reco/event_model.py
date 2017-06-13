# Clsses defining the event model
# sensible.
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

class Hit:
    """Represents a reconstructed hit"""
    def __init__(self):
        self.npeak = -1
        self.X     = -1e12
        self.Y     = -1e12
        self.Z     = -1
        self.E     = -1
        self.Q     = -1
        self.nsipm = -1

    @property
    def R(self): return np.sqrt(self.X ** 2 + self.Y ** 2)

    @property
    def Phi(self): return np.arctan2(self.Y, self.X)

    def __str__(self):
        return """<npeak = {} nsipm = {} Q = {}
                x = {} y = {} z = {} E = {} >""".format(self.__class__.__name__,
                                       self.npeak, self.nsipm, sel.Q,
                                       self.X, sel.Y, sel.Z, self.E)

    __repr__ =     __str__


class HitCollection:
    def __init__(self):
        super().__init__()
        self.hits = []

    def __str__(self):
        s =  "{}".format(self.__class__.__name__)
        s+= "Hit list:"
        s = [s + str(hit) for hit in self.hits]
        return s

    __repr__ =     __str__
