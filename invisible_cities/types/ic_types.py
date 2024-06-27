from enum        import Enum
from collections import OrderedDict

import numpy as np

NN= -999999  # No Number, a trick to aovid nans in data structs

NoneType = type(None)

class NNN:

    def __getattr__(self, _):
        return NN


class xy:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def empty():
        return xy(NN, NN)

    def zero():
        return xy(0, 0)

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
        return 'xy(x={.x}, y={.y})'.format(self, self)
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



types_dict_summary = OrderedDict({'event'     : np.int64  , 'evt_energy' : np.float64, 'evt_charge'    : np.float64,
                                  'evt_ntrks' : int       , 'evt_nhits'  : int       , 'evt_x_avg'     : np.float64,
                                  'evt_y_avg' : np.float64, 'evt_z_avg'  : np.float64, 'evt_r_avg'     : np.float64,
                                  'evt_x_min' : np.float64, 'evt_y_min'  : np.float64, 'evt_z_min'     : np.float64,
                                  'evt_r_min' : np.float64, 'evt_x_max'  : np.float64, 'evt_y_max'     : np.float64,
                                  'evt_z_max' : np.float64, 'evt_r_max'  : np.float64, 'evt_out_of_map': bool      })




types_dict_tracks = OrderedDict({'event'           : np.int64  , 'trackID'       : int       , 'energy'      : np.float64,
                                 'length'          : np.float64, 'numb_of_voxels': int       , 'numb_of_hits': int       ,
                                 'numb_of_tracks'  : int       , 'x_min'         : np.float64, 'y_min'       : np.float64,
                                 'z_min'           : np.float64, 'r_min'         : np.float64, 'x_max'       : np.float64,
                                 'y_max'           : np.float64, 'z_max'         : np.float64, 'r_max'       : np.float64,
                                 'x_ave'           : np.float64, 'y_ave'         : np.float64, 'z_ave'       : np.float64,
                                 'r_ave'           : np.float64, 'extreme1_x'    : np.float64, 'extreme1_y'  : np.float64,
                                 'extreme1_z'      : np.float64, 'extreme2_x'    : np.float64, 'extreme2_y'  : np.float64,
                                 'extreme2_z'      : np.float64, 'blob1_x'       : np.float64, 'blob1_y'     : np.float64,
                                 'blob1_z'         : np.float64, 'blob2_x'       : np.float64, 'blob2_y'     : np.float64,
                                 'blob2_z'         : np.float64, 'eblob1'        : np.float64, 'eblob2'      : np.float64,
                                 'ovlp_blob_energy': np.float64,
                                 'vox_size_x'      : np.float64, 'vox_size_y'    : np.float64, 'vox_size_z'  : np.float64})
