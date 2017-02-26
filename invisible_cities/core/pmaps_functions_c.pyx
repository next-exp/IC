"""
Cython version of PMAPS
JJGC December, 2016
"""

cimport numpy as np
import  numpy as np


cpdef cdf_to_dict(int df_index, long int evt_max,
                  int   [:] df_event,  int [:] df_peak,
                  float [:] df_time, float [:] df_ene):

    """
    Auxiliary (fast) function to transform PMAPS in in pytables-DF format
    into PMAPS in dictionary format
    """
    cdef int evt = 0
    cdef int pk  = 0

    cdef dict S12L = {}
    cdef dict S12  = {}
    S12 [0] = [[],[]]
    S12L[0] = S12

    cdef int i,
    if evt_max < 0:
      evt_max = np.iinfo(np.int32).max

    for i in range(df_index):
        if evt >= evt_max:
            break
        if df_event[i] == evt:
            S12 = S12L[evt]
            if df_peak[i] == pk:

                s12l = S12[pk]
                s12l[0].append(df_time[i])
                s12l[1].append(df_ene [i])
            else:
                pk = df_peak[i]
                S12[pk] = ([df_time[i]], [df_ene[i]])
        else:

            S12 = S12L[evt]
            for j in S12.keys():
                s12l = S12[j]
                t = np.array(s12l[0])
                e = np.array(s12l[1])
                S12[j] = [t,e]
            S12L[evt] = S12

            evt = df_event[i]
            if evt >= evt_max:
                break

            pk = df_peak[i]
            S12 = {}
            S12[pk] = ([df_time[i]], [df_ene[i]])
            S12L[evt] = S12

    return S12L
