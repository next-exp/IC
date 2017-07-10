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
        def __get__(self):  return (False
                                    if np.any(np.isnan(self.t))  or
                                       np.any(np.isnan(self.E))
                                    else True)


    def __str__(self):
        s = """Peak(samples = {0:d} width = {1:8.1f} mus , energy = {2:8.1f} pes
        height = {3:8.1f} pes tmin-tmax = {4} mus """.format(self.number_of_samples,
        self.width / units.mus, self.total_energy, self.height,
        (self.tmin_tmax * (1 / units.mus)))
        return s

    def __repr__(self):
        return self.__str__()


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
    def __init__(self, dict s12d):

        cdef int event_no
        cdef np.ndarray[double, ndim=1] t
        cdef np.ndarray[double, ndim=1] E
        self.peaks = {}

        #print('s12d ={}'.format(s12d))
        for event_no, (t, E) in s12d.items():
            #print('t ={}'.format(t))
            #print('E ={}'.format(E))
            assert len(t) == len(E)
            #p = Peak(t,E)
            #print('peak = {}'.format(p))

            self.peaks[event_no] =  Peak(t, E)

    property number_of_peaks:
         def __get__(self): return len(self.peaks)

    cpdef peak_collection(self):
        try:
            return tuple(self.peaks.keys())
        except KeyError:
            raise PeakNotFound

    cpdef peak_waveform(self, int peak_number):
        try:
            return self.peaks[peak_number]
        except KeyError:
             raise PeakNotFound

    cpdef store(self, table, event_number):
        row = table.row
        for peak_number, peak in self.peaks.items():
            for t, E in zip(peak.t, peak.E):
                row["event"] = event_number
                row["peak"]  =  peak_number
                row["time"]  = t
                row["ene"]   = E
                row.append()


cdef class S1(S12):
    def __init__(self, s1d):
        self.s1d = s1d
        super(S1,self).__init__(s1d)

    def __str__(self):
        s =  "S1 (number of peaks = {})\n".format(self.number_of_peaks)
        s2 = ['peak number = {}: {} \n'.format(i,
                                    self.peak_waveform(i)) for i in self.peaks]
        return  s + ''.join(s2)

    def __repr__(self):
        return self.__str__()

cdef class S2(S12):
    def __init__(self, s2d):
        self.s2d = s2d
        super(S2, self).__init__(s2d)

    def __str__(self):
        s =  "S2 (number of peaks = {})\n".format(self.number_of_peaks)
        s2 = ['peak number = {}: {} \n'.format(i,
                                    self.peak_waveform(i)) for i in self.peaks]
        return  s + ''.join(s2)

    def __repr__(self):
        return self.__str__()

cdef class S2Si(S2):
    """Transient class representing the combination of
    S2 and the SiPM information.
    Notice that S2Si is constructed using an s2d and an s2sid.
    The s2d is an s12 dictionary (not an S2 instance)
    The s2sid is a dictionary {peak:{nsipm:[E]}}
    """

    def __init__(self, s2d, s2sid):
        """where:
           s2d   = {peak_number:[[t], [E]]}
           s2sid = {peak:{nsipm:[Q]}}
           Q is the energy in each SiPM sample
        """
        S2.__init__(self, s2d)
        self.s2sid = s2sid

    cpdef number_of_sipms_in_peak(self, int peak_number):
        return len(self.s2sid[peak_number])

    cpdef sipms_in_peak(self, int peak_number):
        try:
            return tuple(self.s2sid[peak_number].keys())
        except KeyError:
            raise PeakNotFound

    cpdef sipm_waveform(self, int peak_number, int sipm_number):
        cdef double [:] E
        if self.number_of_sipms_in_peak(peak_number) == 0:
            raise SipmEmptyList
        try:
            E = self.s2sid[peak_number][sipm_number]
            #print("in sipm_waveform")
            #print('t ={}'.format(self.peak_waveform(peak_number).t))
            #print('E ={}'.format(np.asarray(E)))
            return Peak(self.peak_waveform(peak_number).t, np.asarray(E))
        except KeyError:
            raise SipmNotFound

    cpdef sipm_waveform_zs(self, int peak_number, int sipm_number):
        cdef double [:] E, t, tzs, Ezs
        cdef list TZS = []
        cdef list EZS = []
        cdef int i
        if self.number_of_sipms_in_peak(peak_number) == 0:
            raise SipmEmptyList("No SiPMs associated to this peak")
        try:
            E = self.s2sid[peak_number][sipm_number]
            t = self.peak_waveform(peak_number).t

            for i in range(len(E)):
                if E[i] > 0:
                    TZS.append(t[i])
                    EZS.append(E[i])
            tzs = np.array(TZS)
            Ezs = np.array(EZS)

            return Peak(np.asarray(tzs), np.asarray(Ezs))
        except KeyError:
            raise SipmNotFound

    cpdef sipm_total_energy(self, int peak_number, int sipm_number):
        """For peak and and sipm_number return Q, where Q is the SiPM total energy."""
        cdef double et
        if self.number_of_sipms_in_peak(peak_number) == 0:
            return 0
        try:
            et = np.sum(self.s2sid[peak_number][sipm_number])
            return et
        except KeyError:
            raise SipmNotFound

    cpdef sipm_total_energy_dict(self, int peak_number):
        """For peak number return {sipm: Q}. """
        cdef dict Q_sipm_dict = {}
        if self.number_of_sipms_in_peak(peak_number) == 0:
            return Q_sipm_dict
        for sipm_number in self.sipms_in_peak(peak_number):
            Q_sipm_dict[sipm_number] = self.sipm_total_energy( peak_number, sipm_number)
        return Q_sipm_dict

    cpdef peak_and_sipm_total_energy_dict(self):
        """Return {peak_no: sipm: Q}."""
        cdef dict Q_dict = {}
        for peak_number in self.peak_collection():
            Q_dict[peak_number] = self.sipm_total_energy_dict(peak_number)

        return Q_dict

    cpdef store(self, table, event_number):
        row = table.row
        for peak, sipm in self.s2sid.items():
            for nsipm, ene in sipm.items():
                for E in ene:
                    row["event"]   = event_number
                    row["peak"]    = peak
                    row["nsipm"]   = nsipm
                    row["ene"]     = E
                    row.append()

    def __str__(self):
        s  = "=" * 80 + "\n" + S2.__str__(self)

        s += "-" * 80 + "\nSiPMs for non-empty peaks\n\n"

        s2a = ["peak number = {}: nsipm in peak = {}"
               .format(peak_number, self.sipms_in_peak(peak_number))
               for peak_number in self.peaks
               if len(self.sipms_in_peak(peak_number)) > 0]

        s += '\n\n'.join(s2a) + "\n"

        s += "-" * 80 + "\nSiPMs Waveforms\n\n"

        s2b = ["peak number = {}: sipm number = {}\n    sipm waveform (zs) = {}".format(peak_number, sipm_number, self.sipm_waveform_zs(peak_number, sipm_number))
               for peak_number in self.peaks
               for sipm_number in self.sipms_in_peak(peak_number)
               if len(self.sipms_in_peak(peak_number)) > 0]

        return s + '\n'.join(s2b) + "\n" + "=" * 80

    def __repr__(self):
        return self.__str__()
