import numpy as np


class XYCorrection:

    def __init__(self, xs, ys, E, U, strategy, ni=None, nj=None):
        if   strategy == 'max'  : E_ref = E.max()
        elif strategy == 'index': E_ref = E[ni,nj]

        dE = E / E_ref if strategy else np.ones_like(E)
        dU = U / E_ref if strategy else np.ones_like(E)

        self._Enorm = { (x,y) : dE[i,j]
                        for i, x in enumerate(xs)
                        for j, y in enumerate(ys) }

        self._Unorm = ({ (x,y) : dU[i,j]
                         for i, x in enumerate(xs)
                         for j, y in enumerate(ys) }

                       if strategy is not None else self._Enorm)

    def E(self, x, y): return self._Enorm[(x,y)]
    def U(self, x, y): return self._Unorm[(x,y)]


def FCorrection(E, z, fn, *args):
    return E * fn(z, *args)
