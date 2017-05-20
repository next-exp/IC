import numpy as np


class XYCorrection:

    def __init__(self, xs, ys, E, U, strategy):
        E_max = E.max()
        dE = E / E_max if strategy else np.ones_like(E)
        dN = U / E_max
        self._Enorm = { (x,y) : dE[i,j]
                        for i, x in enumerate(xs)
                        for j, y in enumerate(ys) }

        self._Unorm = ({ (x,y) : dN[i,j]
                         for i, x in enumerate(xs)
                         for j, y in enumerate(ys) }

                       if strategy else self._Enorm)

    def E(self, x, y): return self._Enorm[(x,y)]
    def U(self, x, y): return self._Unorm[(x,y)]
