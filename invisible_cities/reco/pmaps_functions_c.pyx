"""
Cython version of PMAPS
JJGC December, 2016
"""

cimport numpy as np
import  numpy as np

from .  params     import Peak
from .. io.pmap_io  import S12
from .. io.pmap_io import S2Si


cpdef df_to_pmaps_dict(df, max_events=None):
    """Transform S1/S2 from DF format to dict format."""
    cdef dict all_events = {}
    cdef dict current_event
    cdef tuple current_peak

    cdef int   [:] event = df.event.values
    cdef char  [:] peak  = df.peak .values
    cdef float [:] time  = df.time .values
    cdef float [:] ene   = df.ene  .values

    cdef int df_size = len(df.index)
    cdef int i
    cdef long limit = np.iinfo(int).max if max_events is None or max_events < 0 else max_events
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
            current_event[peak_number] = Peak(np.array(t), np.array(E))

    return S12(all_events)


cpdef df_to_s2si_dict(df, max_events=None):
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
    cdef long limit = np.iinfo(int).max if max_events is None or max_events < 0 else max_events

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

    return S2Si(all_events)
