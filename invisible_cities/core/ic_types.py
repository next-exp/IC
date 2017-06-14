# TODO: move this into a more appropriate module
class minmax:

    def __init__(self, min, max):
        assert min <= max
        self.min = min
        self.max = max

    def __mul__(self, factor):
        return minmax(self.min * factor, self.max * factor)

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
