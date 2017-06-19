
import numpy as np

class xy:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    @property
    def pos(self): return np.stack(([self.x], [self.y]), axis=1)

    @property
    def XY(self): return (self.x, self.y)

    @property
    def X(self): return self.x

    @property
    def Y(self): return self.y

    @property
    def R(self): return np.sqrt(self.x ** 2 + self.y ** 2)

    @property
    def Phi(self): return np.arctan2(self.y, self.x)

    def __str__(self):
        return 'xy(x={.x}, max={.y})'.format(self, self)
    __repr__ = __str__

    def __getitem__(self, n):
        if n == 0: return self.x
        if n == 1: return self.y
        raise IndexError

class minmax:

    def __init__(self, min, max):
        assert min <= max
        self.min = min
        self.max = max

    @property
    def bracket(self): return self.max - self.min

    @property
    def center(self): return (self.max + self.min) / 2

    def __mul__(self, factor):
        return minmax(self.min * factor, self.max * factor)

    def __div__(self, factor):
        return minmax(self.min / factor, self.max / factor)

    def __add__(self, scalar):
        return minmax(self.min + scalar, self.max + scalar)

    def __sub__(self, scalar):
        return minmax(self.min - scalar, self.max - scalar)

    def __str__(self):
        return 'minmax(min={.min}, max={.max})'.format(self, self)
    __repr__ = __str__

    def __getitem__(self, n):
        if n == 0: return self.min
        if n == 1: return self.max
        raise IndexError
