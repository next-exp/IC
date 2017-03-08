"""
Cython version of PMAPS
JJGC December, 2016
"""

cimport numpy as np
import  numpy as np

from invisible_cities.reco.params import Peak

cpdef df_to_pmaps_dict(df, max_events=None):
    cdef dict all_events = {}

    cdef int   [:] event = df.event.values
    cdef char  [:] peak  = df.peak .values
    cdef float [:] time  = df.time .values
    cdef float [:] ene   = df.ene  .values

    cdef int df_size = len(df.index)
    cdef int current_event = -2
    cdef int current_peak  = -1
    cdef int i
    cdef long limit = np.iinfo(int).max if max_events is None or max_events < 0 else max_events
    cdef bint event_boundary = True
    cdef bint  peak_boundary = True

    for i in range(df_size):

        if event_boundary: # Start new event
            current_event = event[i]
            if current_event >= limit: break
            event_data = {}

        if peak_boundary:  # Start new peak
            current_peak = peak[i]
            energies, times = [], []

        # Add energy and time to current peak's data
        energies.append(ene [i])
        times   .append(time[i])

        event_boundary = i+1 == df_size or event[i+1] != current_event
        peak_boundary  = event_boundary or peak [i+1] != current_peak

        if peak_boundary:  # End of peak: add it to this event
            event_data[current_peak] = Peak(np.array(times),
                                            np.array(energies))

        if event_boundary: # End of event: add it in the collection of events
            all_events[current_event] = event_data

    return all_events
