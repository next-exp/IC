cimport numpy as np
import numpy as np
from cpython.object cimport Py_EQ

cdef class xy:
    """Represent a (x,y) number"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    property pos:
        def __get__(self):
            return np.stack(([self.x], [self.y]), axis=1)

    property XY:
        def __get__(self):
            return (self.x, self.y)

    property X:
        def __get__(self):
            return self.x

    property Y:
        def __get__(self):
            return self.y

    property x:
        def __get__(self):
            return self.x

    property y:
        def __get__(self):
            return self.y

    property R:
        def __get__(self):
            return np.sqrt(self.x ** 2 + self.y ** 2)

    property Phi:
        def __get__(self):
            return np.arctan2(self.y, self.x)

    def __str__(self):
        return 'xy(x={.x}, max={.y})'.format(self, self)
    __repr__ = __str__

    def __getitem__(self, n):
        if n == 0: return self.x
        if n == 1: return self.y
        raise IndexError


cdef class minmax:

    def __init__(self, min, max):
        assert min <= max
        self.min = min
        self.max = max

    property min:
        def __get__(self):
            return self.min

    property max:
        def __get__(self):
            return self.max

    property bracket:
        def __get__(self):
            return self.max - self.min

    property center:
        def __get__(self):
            return (self.max + self.min) / 2

    def __add__(self, scalar):
        return minmax(self.min + scalar, self.max + scalar)

    def __mul__(self, factor):
        return minmax(self.min * factor, self.max * factor)

    def __truediv__(self, factor):
        assert factor != 0
        return self.__mul__(1./factor)

    def __sub__(self, scalar):
        return minmax(self.min - scalar, self.max - scalar)

    def __richcmp__(x, y, int opt):
        cdef:
            minmax mm
            double min, max
        mm, y = (x, y) if isinstance(x, minmax) else (y,x)
        min = mm.min
        max = mm.max
        if opt == Py_EQ:
            return mm.min == y.min and mm.max == y.max
        else:
            assert False

    def __str__(self):
            return 'minmax(min={.min}, max={.max})'.format(self, self)
    __repr__ = __str__

    def __getitem__(self, n):
        if n == 0: return self.min
        if n == 1: return self.max
        raise IndexError
