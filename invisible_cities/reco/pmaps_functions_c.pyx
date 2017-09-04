"""
Cython functions providing pmaps io and some extra functionality

1. df_to_s1_dict --> transforms pandas df into {event:s1}
2. df_to_s2_dict --> transforms pandas df into {event:s2}
3. df_to_s2si_dict --> transforms pandas df into {event:s2si}

4. integrate_sipm_charges_in_peak(s2si, peak_number)
Returns (np.array[[nsipm_1 ,     nsipm_2, ...]],
         np.array[[sum(q_1), sum(nsipm_2), ...]])

Last revised, Alejandro Botas, August, 2017.
"""

cimport numpy as np
import  numpy as np

from .. evm.pmaps import S1
from .. evm.pmaps import S2
from .. evm.pmaps import S2Si
from .. evm.pmaps import S1Pmt
from .. evm.pmaps import S2Pmt
from .. core.exceptions import InitializedEmptyPmapObject

cpdef integrate_sipm_charges_in_peak(s2si, int peak_number):
    """Return arrays of nsipm and integrated charges from SiPM dictionary.

    S2Si = {  peak : Si }
      Si = { nsipm : [ q1, q2, ...] }

    Returns (np.array[[nsipm_1 ,     nsipm_2, ...]],
             np.array[[sum(q_1), sum(nsipm_2), ...]])
    """

    cdef long[:] sipms = np.asarray(tuple(s2si.sipm_total_energy_dict(peak_number).keys()))
    cdef double[:] Qs    = np.array(tuple(s2si.sipm_total_energy_dict(peak_number).values()))
    return np.asarray(sipms), np.asarray(Qs)


cpdef df_to_s1_dict(df, int max_events=-1):
    """Takes a table with the persistent representation of pmaps
    (in the form of a pandas data frame) and returns a dict {event:S1}

    """
    cdef dict s1_dict = {}
    cdef dict s12d_dict, s12d
    cdef int event_no

    s12d_dict = df_to_pmaps_dict(df, max_events)  # {event:s12d}

    for event_no, s12d in s12d_dict.items():
        try: s1_dict[event_no] = S1(s12d)
        except InitializedEmptyPmapObject: pass

    return s1_dict


cpdef df_to_s2_dict(df, int max_events=-1):
    """Takes a table with the persistent representation of pmaps
    (in the form of a pandas data frame) and returns a dict {event:S1}

    """
    cdef dict s2_dict = {}
    cdef dict s12d_dict, s12d
    cdef int event_no

    s12d_dict = df_to_pmaps_dict(df, max_events)  # {event:s12d}

    for event_no, s12d in s12d_dict.items():
        try: s2_dict[event_no] = S2(s12d)
        except InitializedEmptyPmapObject: pass

    return s2_dict


cpdef df_to_s1pmt_dict(dfs1, dfpmts, int max_events=-1):
    """Takes a table with the persistent representation of pmaps
    (in the form of a pandas data frame) and returns a dict {event:s1pmt}
    """
    cdef dict s1pmt_dict = {}
    cdef s1pmtd_dict, s1_dict, ipmtd, s1d
    cdef int event_no

    s1_dict     = df_to_s1_dict     (dfs1  , max_events)
    s1pmtd_dict = df_to_s12pmtd_dict(dfpmts, max_events)
    for event_no, ipmtd in s1pmtd_dict.items():
        s1d = s1_dict[event_no].s1d
        try: s1pmt_dict[event_no] = S1Pmt(s1d, ipmtd)
        except InitializedEmptyPmapObject: pass
    return s1pmt_dict


cpdef df_to_s2pmt_dict(dfs2, dfpmts, int max_events=-1):
    """Takes a table with the persistent representation of pmaps
    (in the form of a pandas data frame) and returns a dict {event:s2pmt}
    """
    cdef dict s2pmt_dict = {}
    cdef s2pmtd_dict, s2_dict, ipmtd, s2d
    cdef int event_no

    s2_dict     = df_to_s2_dict     (dfs2  , max_events)
    s2pmtd_dict = df_to_s12pmtd_dict(dfpmts, max_events)
    for event_no, ipmtd in s2pmtd_dict.items():
        s2d = s2_dict[event_no].s2d
        try: s2pmt_dict[event_no] = S2Pmt(s2d, ipmtd)
        except InitializedEmptyPmapObject: pass
    return s2pmt_dict


cpdef df_to_s2si_dict(dfs2, dfsi, int max_events=-1):
    """Takes a table with the persistent representation of pmaps
    (in the form of a pandas data frame) and returns a dict {event:S2Si}

    """
    cdef dict s2si_dict = {}
    cdef s2sid_dict, s2_dict, s2sid, s2d
    cdef int event_no

    s2sid_dict = df_to_s2sid_dict(dfsi, max_events)
    s2_dict    = df_to_s2_dict(dfs2, max_events)

    for event_no, s2sid in s2sid_dict.items():
        s2d = s2_dict[event_no].s2d
        try: s2si_dict[event_no] = S2Si(s2d, s2sid)
        except InitializedEmptyPmapObject: pass

    return s2si_dict

cdef df_to_pmaps_dict(df, int max_events):
    """Takes a table with the persistent representation of pmaps
    (in the form of a pandas data frame) and returns a dict {event:s12d}

    """

    cdef dict all_events = {}
    cdef dict current_event
    cdef tuple current_peak

    cdef int   [:] event = df.event.values
    cdef char  [:] peak  = df.peak .values
    cdef float [:] time  = df.time .values
    cdef float [:] ene   = df.ene  .values

    cdef int df_size = len(df.index)
    cdef int i
    cdef long limit = np.iinfo(int).max if max_events < 0 else max_events
    cdef int peak_number
    cdef list t, E

    for i in range(df_size):
        if event[i] >= limit: break

        current_event = all_events   .setdefault(event[i], {}      )
        current_peak  = current_event.setdefault( peak[i], ([], []))
        current_peak[0].append(time[i])
        current_peak[1].append( ene[i])

    # Postprocessing: Turn lists to numpy arrays before returning
    for current_event in all_events.values():
        for peak_number, (t, E) in current_event.items():
            current_event[peak_number] = [np.array(t), np.array(E)]

    return all_events


cdef df_to_s12pmtd_dict(df, int max_events):
    """ Transform S2Si from DF format to dict format."""
    cdef dict all_events = {}
    cdef dict current_event
    cdef list current_peak
    cdef int   [:] event = df.event.values
    cdef char  [:] peak  = df.peak .values
    cdef char  [:] npmt  = df.npmt .values
    cdef float [:] ene   = df.ene  .values
    cdef int  df_size        = len(df.index)
    cdef int  number_of_pmts = len(set(npmt))
    cdef long limit = np.iinfo(int).max if max_events < 0 else max_events

    cdef int  i, j
    for i in range(df_size):
        if event[i] >= limit: break
        current_event = all_events   .setdefault(event[i], {} )
        current_peak  = current_event.setdefault( peak[i], [[] for j in range(number_of_pmts)])
        current_peak[npmt[i]].append(ene[i])

    # Postprocessing: Turn lists to numpy arrays before returning and fill
    # empty slices with zeros
    cdef int  pn
    cdef list energy
    for current_event in all_events.values():
        for pn, energy in current_event.items():
            current_event[pn] = np.array(energy)

    return all_events


cdef df_to_s2sid_dict(df, int max_events):
    """ Transform S2Si from DF format to dict format."""
    cdef dict all_events = {}
    cdef dict current_event
    cdef dict current_peak
    cdef list current_sipm

    cdef int   [:] event = df.event.values
    cdef char  [:] peak  = df.peak .values
    cdef short [:] nsipm = df.nsipm.values
    cdef float [:] ene   = df.ene  .values

    cdef int df_size = len(df.index)
    cdef long limit = np.iinfo(int).max if max_events < 0 else max_events

    cdef int i
    for i in range(df_size):
        if event[i] >= limit: break

        current_event = all_events   .setdefault(event[i], {} )
        current_peak  = current_event.setdefault( peak[i], {} )
        current_sipm  = current_peak .setdefault(nsipm[i], [] )
        current_sipm.append(ene[i])

    cdef int  ID
    cdef list energy
    # Postprocessing: Turn lists to numpy arrays before returning and fill
    # empty slices with zeros
    for current_event in all_events.values():
        for current_peak in current_event.values():
            for ID, energy in current_peak.items():
                current_peak[ID] = np.array(energy)

    return all_events


cpdef sipm_ids_and_charges_in_slice(dict s2sid_peak, int slice_no):
    """Given s2sid_peak = {nsipm : [ q1, q2, ...qn]} and a slice_no
    (running from 1, 2..n) returns:
    Returns (np.array[nsipm_1 , nsipm_2, ...],
             np.array[q_k from nsipm_1, q_k from nsipm_2, ...]]) when slice_no=k
    """

    cdef int number_of_sipms = len(s2sid_peak.keys())
    cdef list ids = []
    cdef list qs_slice = []

    cdef short int i, nsipm
    for i, (nsipm, qs) in enumerate(s2sid_peak.items()):
        if qs[slice_no] > 0:
            ids.append(nsipm)
            qs_slice.append(qs[slice_no])

    return np.array(ids, dtype=np.int), np.array(qs_slice, dtype=np.double)


cpdef _impose_thr_sipm_destructive(dict s2si_dict, float thr_sipm):
    """imposes a thr_sipm on s2si_dict"""
    cdef dict si_peak
    cdef int sipm, i
    cdef float q
    for s2si in s2si_dict.values():                   # iter over events
        for si_peak in s2si.s2sid.values():           # iter over peaks
            for sipm in list(si_peak.keys()):         # iter over sipms ** avoid mod while iter
                for i, q in enumerate(si_peak[sipm]): # iter over timebins
                    if q < thr_sipm:                  # impose threshold
                        si_peak[sipm][i] = 0
                if si_peak[sipm].sum() == 0:          # Delete SiPMs with integral
                    del si_peak[sipm]                 # charge equal to 0
    return s2si_dict


cpdef _impose_thr_sipm_s2_destructive(dict s2si_dict, float thr_sipm_s2):
    """imposes a thr_sipm_s2 on s2si_dict. deletes keys (sipms) from each s2sid peak if sipm
       integral charge is less than thr_sipm_s2"""
    cdef dict si_peak
    cdef int sipm
    cdef np.ndarray qs
    for s2si in s2si_dict.values():
        for si_peak in s2si.s2sid.values():
            for sipm, qs in list(si_peak.items()): # ** avoid modifying while iterating
                sipm_integral_charge = qs.sum()
                if sipm_integral_charge < thr_sipm_s2:
                    del si_peak[sipm]
    return s2si_dict


cpdef _delete_empty_s2si_peaks(dict s2si_dict):
    """makes sure there are no empty peaks stored in an s2sid
        (s2sid[pn] != {} for all pn in s2sid and all s2sid in s2si_dict)
        ** Also deletes corresponding peak in s2si.s2d! """
    cdef int ev, pn
    for ev in list(s2si_dict.keys()):
        for pn in list(s2si_dict[ev].s2sid.keys()):
            if len(s2si_dict[ev].s2sid[pn]) == 0:
                del s2si_dict[ev].s2sid[pn]
                del s2si_dict[ev].s2d  [pn]
                # It is not sufficient to just delete the peaks because the S2Si class instance
                # will still think it has peak pn even though its base dictionary does not
                try: s2si_dict[ev] = S2Si(s2si_dict[ev].s2d, s2si_dict[ev].s2sid)
                except InitializedEmptyPmapObject: del s2si_dict[ev]
    return s2si_dict


cpdef _delete_empty_s2si_dict_events(dict s2si_dict):
    """ delete all events from s2si_dict with empty s2sid"""
    cdef int ev
    for ev in list(s2si_dict.keys()):
        if len(s2si_dict[ev].s2sid) == 0:
            del s2si_dict[ev]
    return s2si_dict
