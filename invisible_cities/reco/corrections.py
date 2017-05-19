import numpy as np

def Correction(xs, ys, E, _, strategy):
    N = E / E.max() if strategy else np.ones_like(E)
    norm = { (x,y) : N[i,j]
             for i, x in enumerate(xs)
             for j, y in enumerate(ys) }

    def correct(x, y):
        return norm[(x,y)]

    return correct
