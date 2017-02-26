"""
Cython version of PMAPS
JJGC December, 2016
"""
cimport numpy as np
import numpy as np


cpdef cdf_to_dict(int df_index, long int evt_max,
                  int [:] df_event, int [:] df_peak,
                  float [:] df_time, float [:] df_ene)
