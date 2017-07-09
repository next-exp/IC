"""
Cython version of PMAPS
JJGC December, 2016
"""
cimport numpy as np
import numpy as np

"""Takes a table with the persistent representation of pmaps
(in the form of a pandas data frame) and returns a dict {event:s12d}
"""
#cpdef df_to_pmaps_dict(df, max_events=?)

"""Takes a table with the persistent representation of pmaps
(in the form of a pandas data frame) and returns a dict {event:S2Si} """
cpdef df_to_s2si_dict(dfs2, dfsi, int max_events=*)

"""Takes a table with the persistent representation of pmaps
(in the form of a pandas data frame) and returns a dict {event:S1} """
cpdef df_to_s1_dict(df, int max_events=*)

"""Takes a table with the persistent representation of pmaps
(in the form of a pandas data frame) and returns a dict {event:S2} """
cpdef df_to_s2_dict(df, int max_events=*)
