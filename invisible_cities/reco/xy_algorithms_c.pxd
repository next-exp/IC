cimport numpy as np
import numpy as np

"""
pos = column np.array --> (matrix n x 2)
   ([x1, y1],
    [x2, y2]
    ...
    [xs, ys])
   qs = vector (q1, q2...qs) --> (1xn)
"""
cpdef barycenter(np.ndarray[double, ndim=2] pos, np.ndarray[double, ndim=1] qs)


"""
corona creates a list of Clusters by
first , identifying hottest_sipm, the sipm with max charge in qs (must be > Qlm)
second, calling barycenter() on the pos and qs SiPMs within lm_radius of hottest_sipm to
        find new_local_maximum.
third , calling barycenter() on all SiPMs within new_lm_radius of new_local_maximum
fourth, recording the Cluster found by barycenter if the cluster contains at least msipm
fifth , removing (nondestructively) the sipms contributing to that Cluster
sixth , repeating 1-5 until there are no more SiPMs of charge > Qlm
"""
cpdef corona(np.ndarray[double, ndim=2] pos, np.ndarray[double, ndim=1] qs,
             float Qthr           =  *,
             float Qlm            =  *,
             float lm_radius      =  *,
             float new_lm_radius  =  *,
             int msipm            =  *)


cpdef discard_sipms(np.ndarray[  long, ndim=1] sis,
                    np.ndarray[double, ndim=2] pos,
                    np.ndarray[double, ndim=1] qs)

"""return indices of sipms less than d from (xc,yc)"""
cpdef get_nearby_sipm_inds(tuple cs, float d,
                           np.ndarray[double, ndim=2] pos,
                           np.ndarray[double, ndim=1] qs)
