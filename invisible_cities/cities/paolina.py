from functools import reduce

from collections import namedtuple

import numpy as np

class Hit:

    def __init__(self, x,y,z, E):
        self.pos = np.array((x,y,z))
        self.E   = E

    def __str__(self):
        return '<{} {} {}>'.format(self.__class__.__name__,
                                   self.pos.tolist(), self.E)
    __repr__ = __str__

class Voxel:

    def __init__(self, x,y,z, E):
        self.pos = np.array((x,y,z))
        self.E   = E

    __str__  = Hit.__str__
    __repr__ =     __str__

MAX3D = np.array([float(' inf')] * 3)
MIN3D = np.array([float('-inf')] * 3)

def bounding_box(seq):
    posns = [x.pos for x in seq]
    return (reduce(np.minimum, posns, MAX3D),
            reduce(np.maximum, posns, MIN3D))


def voxelize_hits(hits, voxel_dimensions):
    return [Voxel(*hits[0].pos, h.E) for h in hits]
