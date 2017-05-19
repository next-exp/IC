import numpy as np


class Correction:

    def __init__(self, xs, ys, E, _, strategy):
        N = E / E.max() if strategy else np.ones_like(E)
        self._norm = { (x,y) : N[i,j]
                       for i, x in enumerate(xs)
                       for j, y in enumerate(ys) }

    def E(self, x, y):
        return self._norm[(x,y)]

    def U(self, x, y):
        return self._norm[(x,y)]
