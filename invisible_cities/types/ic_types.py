
import numpy as np

class Counter:

    def __init__(self, counter_name=''):
        self.cd = {}
        self.cn = counter_name

    def init_counter(self, name, value=0):
        self.cd.setdefault(name, value)

    def init_counters(self, name_list, value=None):
        if value==None:
            for name in name_list:
                self.cd.setdefault(name, 0)
        else:
            for value_no, name in enumerate(name_list):
                self.cd.setdefault(name, value[value_no])

    def increment_counter(self, name, value=1):
        self.cd[name] += value

    def increment_counters(self, name_list, value=None):
        if value==None:
            for name in name_list:
                self.cd[name] += 1
        else:
            for value_no, name in enumerate(name_list):
                self.cd[name] += value[value_no]

    def set_counter(self, name, value=0):
        self.cd[name] = value

    def set_counters(self, name_list, value=None):
        if value==None:
            for name in name_list:
                self.cd[name] =0
        else:
            for value_no, name in enumerate(name_list):
                self.cd[name] = value[value_no]

    def counter_value(self, name):
        return self.cd[name]

    def counters(self):
        return tuple(self.cd.keys())

    def __str__(self):
        s = self.cn+':'
        s2 = [' (counter = {}, value = {}), '.format(counter, value) for counter, value in self.cd.items()]
        return  s + ''.join(s2)

    __repr__ = __str__


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
