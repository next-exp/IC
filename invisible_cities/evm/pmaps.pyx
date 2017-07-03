# Clsses defining the event model

cimport numpy as np
import numpy as np

from .. core.ic_types_c        cimport minmax
from .. core.exceptions        import PeakNotFound
from .. core.exceptions        import SipmEmptyList
from .. core.exceptions        import SipmNotFound
from .. core.core_functions    import loc_elem_1d
from .. core.system_of_units_c import units


cdef class Peak:
    """Transient class representing a Peak.

    A Peak is represented as a pair of arrays:
    t: np.array() describing time bins
    E: np.array() describing energy.
    """

    def __init__(self, np.ndarray[double, ndim=1] t,
                       np.ndarray[double, ndim=1] E):

        cdef int i_t
        assert len(t) == len(E)
        self.t              = t
        self.E              = E
        self.height         = np.max(self.E)
        self.width          = self.t[-1] - self.t[0]
        self.total_energy   = np.sum(self.E)

        i_t    = (loc_elem_1d(self.E, self.height)
                             if self.total_energy > 0
                             else 0)

        self.tpeak  =  self.t[i_t]


    property tmin_tmax:
        def __get__(self): return minmax(self.t[0], self.t[-1])

    property number_of_samples:
        def __get__(self): return len(self.t)


    property good_waveform:
        def __get__(self):  return (True
                                    if np.any(np.isnan(self.t))  or
                                       np.any(np.isnan(self.E))
                                    else False)


    def __str__(self):
        s = """Peak(samples = {} width = {:.1f} mus , energy = {:.1f} pes
        height = {:.1f} pes tmin-tmax = {} mus """.format(self.number_of_samples,
        self.width / units.mus, self.total_energy, self.height,
        (self.tmin_tmax * (1 / units.mus)).__str__(1))
        return s

    __repr__ = __str__


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
    def __init__(self, s12d):
        self.peaks = {i: Peak(t, E) for i, (t,E) in s12d.items()}

    property number_of_peaks:
         def __get__(self): return len(self.peaks)

    cpdef peak_waveform(self, peak_number):
        try:
            return self.peaks[peak_number]
        except KeyError:
             raise PeakNotFound

    def __str__(self):
        s =  "{}(number of peaks = {})\n".format(self.__class__.__name__, self.number_of_peaks)
        s2 = ['peak number = {}: {} \n'.format(i,
                                    self.peak_waveform(i)) for i in self.peaks]
        return  s + ''.join(s2)
    __repr__ = __str__
