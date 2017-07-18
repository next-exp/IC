"""
Cython version of PMAPS
JJGC December, 2016
"""
cimport numpy as np
import numpy as np

"""Return arrays of nsipm and integrated charges from S2Si.

Returns (np.array[[nsipm_1 ,     nsipm_2, ...]],
         np.array[[sum(q_1), sum(nsipm_2), ...]])
"""

cpdef integrate_sipm_charges_in_peak(s2si, int peak_number)

"""Takes a table with the persistent representation of pmaps
(in the form of a pandas data frame) and returns a dict {event:S2Si} """
cpdef df_to_s2si_dict(dfs2, dfsi, int max_events=*)

"""Takes a table with the persistent representation of pmaps
(in the form of a pandas data frame) and returns a dict {event:S1} """
cpdef df_to_s1_dict(df, int max_events=*)

"""Takes a table with the persistent representation of pmaps
(in the form of a pandas data frame) and returns a dict {event:S2} """
cpdef df_to_s2_dict(df, int max_events=*)

"""Given s2sid_peak = {nsipm : [ q1, q2, ...qn]} and a slice_no
(running from 1, 2..n) returns:
Returns (np.array[nsipm_1 , nsipm_2, ...],
         np.array[q_k from nsipm_1, q_k from nsipm_2, ...]]) when slice_no=k
 """
cpdef sipm_ids_and_charges_in_slice(dict s2sid_peak, int slice_no)
