
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
    cpdef peak_collection(self)
    cpdef peak_waveform(self, int peak_number)
    cpdef store(self, table, event_number)


cdef class S1(S12):
    """Transient class representing an S1 signal."""
    cdef public s1d


cdef class S2(S12):
    """Transient class representing an S2 signal."""
    cdef public s2d


cdef class S2Si(S2):
    """Transient class representing the combination of
    S2 and the SiPM information.
    Notice that S2Si is constructed using an s2d and an s2sid.
    The s2d is an s12 dictionary (not an S2 instance)
    The s2sid is a dictionary {peak:{nsipm:[E]}}
    """
    cdef public s2sid
    cdef dict _s2sid
    cpdef number_of_sipms_in_peak(self, int peak_number)
    cpdef sipms_in_peak(self, int peak_number)
    cpdef sipm_waveform(self, int peak_number, int sipm_number)
    cpdef sipm_waveform_zs(self, int peak_number, int sipm_number)
    cpdef sipm_total_energy(self, int peak_number, int sipm_number)
    cpdef sipm_total_energy_dict(self, int peak_number)
    cpdef peak_and_sipm_total_energy_dict(self)
    cpdef store(self, table, event_number)

"""Given an s2d and an s2sid, return an s2d containing only the peaks shared by s2sid"""
cpdef check_s2d_and_s2sid_share_peaks(dict s2d, dict s2sid)
