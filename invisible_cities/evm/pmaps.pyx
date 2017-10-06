# Clsses defining the event model

cimport numpy as np
import numpy as np

from textwrap import dedent

from .. types.ic_types_c       cimport minmax
from .. core.exceptions        import PeakNotFound
from .. core.exceptions        import SipmEmptyList
from .. core.exceptions        import SipmNotFound
from .. core.exceptions        import PmtNotFound
from .. core.core_functions    import loc_elem_1d
from .. core.exceptions        import InconsistentS12dIpmtd
from .. core.exceptions        import InitializedEmptyPmapObject
from .. core.system_of_units_c import units


cdef class Peak:
    """Transient class representing a Peak.

    A Peak is represented as a pair of arrays:
    t: np.array() describing time bins
    E: np.array() describing energy.
    """

    def __init__(self, np.ndarray[double, ndim=1] t,
                       np.ndarray[double, ndim=1] E):

        if len(t) == 0 or len(E) == 0: raise InitializedEmptyPmapObject
        cdef int i_t
        assert len(t) == len(E)
        self.t              = t
        self.E              = E
        self.height         = np.max(self.E)
        self.width          = self.t[-1] - self.t[0]
        self.total_energy   = np.sum(self.E)

        i_t        = np.argmax(self.E)
        self.tpeak = self.t[i_t]

    property tmin_tmax:
        def __get__(self): return minmax(self.t[0], self.t[-1])

    property number_of_samples:
        def __get__(self): return len(self.t)

    property good_waveform:
        def __get__(self):  return not np.any(np.isnan(self.t) | np.isnan(self.E))

    def signal_above_threshold(self, thr):
        return self.E > thr

    def total_energy_above_threshold(self, thr):
        sat = self.signal_above_threshold(thr)
        return np.sum(self.E[sat])

    def width_above_threshold(self, thr):
        sat = self.signal_above_threshold(thr)
        t   = self.t[sat]
        return t[-1] - t[0] if sat.any() else 0

    def height_above_threshold(self, thr):
        sat = self.signal_above_threshold(thr)
        return np.max(self.E[sat]) if sat.any() else 0

    def __str__(self):
        if self.width < units.mus:
            width   = "{:d} ns".format(self.width)
            tminmax = "{} ns".format(self.tmin_tmax)
        else:
            width   = "{:.1f} mus".format(self.width/units.mus)
            tminmax = "{} mus".format(self.tmin_tmax / units.mus)
        
        return dedent("""
                      Peak(
                           samples   = {self.number_of_samples:d}
                           width     = {width}
                           energy    = {self.total_energy:8.1f} pes
                           height    = {self.height:8.1f} pes
                           tmin-tmax = {tminmax} mus
                      """.format(self = self, width = width, tminmax = tminmax)
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

        if len(s12d) == 0: raise InitializedEmptyPmapObject
        cdef int peak_no
        cdef np.ndarray[double, ndim=1] t
        cdef np.ndarray[double, ndim=1] E
        self.peaks = {}

        for peak_no, (t, E) in s12d.items():
            assert len(t) == len(E)
            self.peaks[peak_no] =  Peak(t, E)

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

        if len(s1d) == 0: raise InitializedEmptyPmapObject
        self.s1d = s1d
        super(S1, self).__init__(s1d)

    def __str__(self):
        s =  "S1 (number of peaks = {})\n".format(self.number_of_peaks)
        s2 = ['peak number = {}: {} \n'.format(i,
                                    self.peak_waveform(i)) for i in self.peaks]
        return  s + ''.join(s2)

    def __repr__(self):
        return self.__str__()


cdef class S2(S12):

    def __init__(self, s2d):
        if len(s2d) == 0: raise InitializedEmptyPmapObject
        self.s2d = s2d
        super(S2, self).__init__(s2d)

    def __str__(self):
        s =  "S2 (number of peaks = {})\n".format(self.number_of_peaks)
        s2 = ['peak number = {}: {} \n'.format(i,
                                    self.peak_waveform(i)) for i in self.peaks]
        return  s + ''.join(s2)

    def __repr__(self):
        return self.__str__()


cpdef check_s2d_and_s2sid_share_peaks(dict s2d, dict s2sid):
    cdef dict s2d_shared_peaks = {}
    cdef int pn
    for pn, peak in s2d.items():
        if pn in s2sid:
          s2d_shared_peaks[pn] = peak

    return s2d_shared_peaks


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

        if len(s2sid) == 0 or len(s2d) == 0: raise InitializedEmptyPmapObject
        S2.__init__(self, check_s2d_and_s2sid_share_peaks(s2d, s2sid))
        self.s2sid = s2sid

    cpdef sipm_peak(self, int peak_number):
        try:
            return self.s2sid[peak_number]
        except KeyError:
            raise PeakNotFound

    cpdef find_sipm(self, int peak_number, int sipm_number):
        try:
            return self.sipm_peak(peak_number)[sipm_number]
        except KeyError:
            raise SipmNotFound

    cpdef number_of_sipms_in_peak(self, int peak_number):
        return len(self.sipm_peak(peak_number))

    cpdef sipms_in_peak(self, int peak_number):
        return tuple(self.sipm_peak(peak_number).keys())

    cpdef sipm_waveform(self, int peak_number, int sipm_number):
        cdef double [:] E
        if self.number_of_sipms_in_peak(peak_number) == 0:
            raise SipmEmptyList

        E = self.find_sipm(peak_number, sipm_number)
        return Peak(self.peak_waveform(peak_number).t, np.asarray(E))

    cpdef sipm_waveform_zs(self, int peak_number, int sipm_number):
        cdef double [:] E, t, tzs, Ezs
        cdef list TZS = []
        cdef list EZS = []
        cdef int i
        if self.number_of_sipms_in_peak(peak_number) == 0:
            raise SipmEmptyList("No SiPMs associated to this peak")

        E = self.find_sipm(peak_number, sipm_number)
        t = self.peak_waveform(peak_number).t

        for i in range(len(E)):
            if E[i] > 0:
                TZS.append(t[i])
                EZS.append(E[i])
        tzs = np.array(TZS)
        Ezs = np.array(EZS)

        return Peak(np.asarray(tzs), np.asarray(Ezs))

    cpdef sipm_total_energy(self, int peak_number, int sipm_number):
        """For peak and and sipm_number return Q, where Q is the SiPM total energy."""
        return np.sum(self.find_sipm(peak_number, sipm_number))

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


cdef class S12Pmt(S12):
    """
    A pmt S12 class for storing individual pmt s12 responses.

    It is analagous to S2Si with the caveat that each peak key in ipmtd maps to a nparray of
    pmt energies instead of another dictionary. Here a dictionary mapping pmt_number --> energy is
    superfluous since the csum of all active pmts are used to calculate the s12 energy.
    """
    def __init__(self, s12d, ipmtd):
        """where:
        s12d  = { peak_number: [[t], [E]]}
        ipmtd = { peak_number: [[Epmt0], [Epmt1], ... ,[EpmtN]] }
        """

        if len(ipmtd) == 0 or len(s12d) == 0: raise InitializedEmptyPmapObject
        # Check that energies in s12d are sum of ipmtd across pmts for each peak
        for peak, s12_pmts in zip(s12d.values(), ipmtd.values()):
            if not np.allclose(peak[1], s12_pmts.sum(axis=0), atol=.01):
                raise InconsistentS12dIpmtd

        S12.__init__(self, s12d)
        self.ipmtd = ipmtd
        cdef int npmts
        try                 : self.npmts = len(ipmtd[next(iter(ipmtd))])
        except StopIteration: self.npmts = 0

    cpdef ipmt_peak(self, int peak_number):
        try:
            return self.ipmtd[peak_number]
        except KeyError:
            raise PeakNotFound

    cpdef find_ipmt(self, int peak_number, int pmt_number):
        try:
            return self.ipmt_peak(peak_number)[pmt_number].astype(np.float64)
        except KeyError:
            raise PmtNotFound

    cpdef energies_in_peak(self, int peak_number):
        return np.asarray(self.ipmt_peak(peak_number))

    cpdef pmt_waveform(self, int peak_number, int pmt_number):
        cdef double [:] E

        E = self.find_ipmt(peak_number, pmt_number)
        return Peak(self.peak_waveform(peak_number).t, np.asarray(E))

    cpdef pmt_total_energy_in_peak(self, int peak_number, int pmt_number):
        """
        For peak_number and and pmt_number return the integrated energy in that pmt in that peak
        ipmtd[peak_number][pmt_number].sum().
        """
        return np.sum(self.find_ipmt(peak_number, pmt_number))

    cpdef pmt_total_energy(self, int pmt_number):
        """
        return the integrated energy in that pmt across all peaks and time bins
        """
        cdef double sum = 0
        cdef int pn
        for pn in self.ipmtd:
            sum += self.pmt_total_energy_in_peak(pn, pmt_number)
        return sum

    def __str__(self):
        ss =  "number of peaks = {}\n".format(self.number_of_peaks)
        ss += "-" * 80 + "\n\n"
        for peak_number in self.peaks:
            s2 = 'peak number = {}: peak waveform for csum ={} \n'.format(peak_number,
                                self.peak_waveform(peak_number))
            s3 = ['pmt number = {}: pmt waveforms = {}\n'.format(pmt_number,
                                    self.pmt_waveform(peak_number, pmt_number))
                                    for pmt_number in range(self.npmts)]
            ss+= s2 + '\n'.join(s3)
        ss += "-" * 80 + "\n\n"
        return ss


    cpdef store(self, table, event_number):
        row = table.row
        for peak, s12_pmts in self.ipmtd.items():
            for npmt, s12_pmt in enumerate(s12_pmts):
                for E in s12_pmt:
                    row["event"]   = event_number
                    row["peak"]    = peak
                    row["npmt"]    = npmt
                    row["ene"]     = E
                    row.append()


cdef class S1Pmt(S12Pmt):
    def __init__(self, s1d, ipmtd):

        if len(ipmtd) == 0 or len(s1d) == 0: raise InitializedEmptyPmapObject
        self.s1d = s1d
        S12Pmt.__init__(self, s1d, ipmtd)

    def __str__(self):
        return "S1Pmt: " + S12Pmt.__str__(self)

    def __repr__(self):
        return self.__str__()



cdef class S2Pmt(S12Pmt):
    def __init__(self, s2d, ipmtd):

        if len(ipmtd) == 0 or len(s2d) == 0: raise InitializedEmptyPmapObject
        self.s2d = s2d
        S12Pmt.__init__(self, s2d, ipmtd)

    def __str__(self):
        return "S2Pmt: " + S12Pmt.__str__(self)

    def __repr__(self):
        return self.__str__()
