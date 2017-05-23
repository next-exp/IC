import numpy as np
from scipy.interpolate import interp2d

class XYCorrection:

    def __init__(self, xs, ys, E, U, strategy, ni=None, nj=None):
        Et, Ut = E.T, U.T
        if   strategy == 'max'  : E_ref = Et.max()
        elif strategy == 'index': E_ref = Et[nj,ni]

        self.E = interp2d(xs, ys, (Et / E_ref) if strategy else np.ones_like(Et))
        self.U = interp2d(xs, ys, (Ut / E_ref) if strategy else np.ones_like(Et))


def FCorrection(E, z, fn, *args):
    return E * fn(z, *args)
