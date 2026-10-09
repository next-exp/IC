from enum        import Enum
from dataclasses import dataclass

from typing  import Union
from typing  import Tuple

import numpy as np

NN= -999999  # No Number, a trick to aovid nans in data structs

NoneType = type(None)

Tuple2Dor3D = Union[Tuple[float, float], Tuple[float, float, float]]


@dataclass(frozen=True)
class Blob:
    energy  : float
    position: np.ndarray
    hit_ids : np.ndarray


class minmax:

    def __init__(self, min, max):
        assert min <= max
        self.min = min
        self.max = max

    @property
    def bracket(self): return self.max - self.min

    @property
    def interval(self): return (self.min, self.max)

    @property
    def center(self): return (self.max + self.min) / 2

    def contains(self, x):
        return self.min <= x <= self.max

    def __mul__(self, factor):
        return minmax(self.min * factor, self.max * factor)

    def __truediv__(self, factor):
        assert factor != 0
        return self.__mul__(1./factor)

    def __add__(self, scalar):
        return minmax(self.min + scalar, self.max + scalar)

    def __sub__(self, scalar):
        return minmax(self.min - scalar, self.max - scalar)

    def __eq__(self, other):
        return self.min == other.min and self.max == other.max

    def __str__(self, decimals=None):
        if decimals is None:
            return 'minmax(min={.min}, max={.max})'.format(self, self)
        fmt = 'minmax(min={{.min:.{0}f}}, max={{.max:.{0}f}})'.format(decimals)
        return fmt.format(self, self)
    __repr__ = __str__

    def __getitem__(self, n):
        if n == 0: return self.min
        if n == 1: return self.max
        raise IndexError


class AutoNameEnumBase(Enum):
    """Automatically generate Enum values from their names.

        Use this as a base class to make Enums with values which automatically
        match the member names:

        class Direction(AutoNameEnumBase):
        ...     LEFT  = auto()
        ...     RIGHT = auto()
        ...
        >>> list(Direction)
        [<Direction.LEFT: 'LEFT'>, <Direction.RIGHT: 'RIGHT'>]
    """
    def _generate_next_value_(name, start, count, last_values):
        return name
