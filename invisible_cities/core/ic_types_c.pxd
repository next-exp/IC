
cdef class xy:
    """Represent a (x,y) number"""
    cdef double x,y

cdef class minmax:
    """Represents a bracketed interval"""
    cdef double min, max
