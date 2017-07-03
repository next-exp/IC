
cdef class Peak:
    """Transient class representing a Peak.

    A Peak is represented as a pair of arrays:
    t: np.array() describing time bins
    E: np.array() describing energy.
    """
    cdef public object t, E
    cdef public double height, width, total_energy, tpeak

cdef class S12:
    """Base class representing an S1/S2 signal
    The S12 attribute is a dictionary s12
    {i: Peak(t,E)}, where i is peak number.
    The notation _s12 is intended to make this
    class private (public classes s1 and s2 will
    extend it).
    The rationale to use s1 and s2 rather than a single
    class s12 to represent both s1 and s2 is that, although
    structurally identical, s1 and s2 represent quite
    different objects. In Particular an s2si is constructed
    with a s2 not a s1.
    An s12 is represented as a dictinary of Peaks.
    """
    cdef public dict peaks
    cpdef peak_waveform(self, peak_number)


# These types merely serve to distinguish the different meanings of
# isomorphic data structures.
cdef class S1(S12):
    """Transient class representing an S1 signal."""
cdef class S2(S12):
    """Transient class representing an S2 signal."""
