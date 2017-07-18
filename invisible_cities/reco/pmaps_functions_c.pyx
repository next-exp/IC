"""
Cython functions providing pmaps io and some extra functionality

1. df_to_s1_dict --> transforms pandas df into {event:s1}
2. df_to_s2_dict --> transforms pandas df into {event:s2}
3. df_to_s2si_dict --> transforms pandas df into {event:s2si}

4. integrate_sipm_charges_in_peak(s2si, peak_number)
Returns (np.array[[nsipm_1 ,     nsipm_2, ...]],
         np.array[[sum(q_1), sum(nsipm_2), ...]])

Last revised, JJGC, July, 2017.
"""

cimport numpy as np
import  numpy as np

# from .  params     import Peak
# from .. io.pmap_io  import S12
# from .. io.pmap_io import S2Si
from .. evm.pmaps import S1
from .. evm.pmaps import S2
from .. evm.pmaps import S2Si

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
        s1_dict[event_no] = S1(s12d)

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
        s2_dict[event_no] = S2(s12d)

    return s2_dict

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
        s2si_dict[event_no] = S2Si(s2d, s2sid)

    return s2si_dict

cdef df_to_pmaps_dict(df, max_events):
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


cdef df_to_s2sid_dict(df, max_events):
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
    cdef list sample
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
    cdef short int  [:] ids      = np.empty(number_of_sipms, dtype=np.int16  )
    cdef float      [:] qs_slice = np.empty(number_of_sipms, dtype=np.float32)

    cdef short int i, nsipm
    cdef double [:] qs
    for i, (nsipm, qs) in enumerate(s2sid_peak.items()):
        if qs[slice_no] > 0:
            ids     [i] = nsipm
            qs_slice[i] = qs[slice_no]

    return np.asarray(ids), np.asarray(qs_slice)
