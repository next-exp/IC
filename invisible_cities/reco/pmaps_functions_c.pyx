"""
Cython version of PMAPS
JJGC December, 2016
"""

cimport numpy as np
import  numpy as np

from invisible_cities.reco.params import Peak

cpdef df_to_pmaps_dict(df, max_events=None):
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

    return all_events


cpdef dict df_to_s2si_dict(df, max_events=None):
    cdef dict  all_events = {}
    cdef dict  current_event
    cdef dict  current_peak
    cdef tuple current_sipm

    cdef int   [:] event   = df.event  .values
    cdef char  [:] peak    = df.peak   .values
    cdef short [:] nsipm   = df.nsipm  .values
    cdef short [:] nsample = df.nsample.values
    cdef float [:] ene     = df.ene    .values

    cdef int  df_size = len(df.index)
    cdef long limit = np.iinfo(int).max if max_events is None or max_events < 0 else max_events

    cdef int i
    for i in range(df_size):
        if event[i] >= limit: break

        current_event = all_events   .setdefault(event[i],  {}     )
        current_peak  = current_event.setdefault( peak[i],  {}     )
        current_sipm  = current_peak .setdefault(nsipm[i], ([], []))
        current_sipm[0].append(nsample[i])
        current_sipm[1].append(    ene[i])

    cdef int  ID
    cdef list sample
    cdef list energy
    # Postprocessing: Turn lists to numpy arrays before returning and fill
    # empty slices with zeros
    for current_event in all_events.values():
        for current_peak in current_event.values():
            for ID, (sample, energy) in current_peak.items():
                maxsample                = np.max(sample) + 1
                current_peak[ID]         = np.zeros(maxsample)
                current_peak[ID][sample] = energy

    return all_events
