
cimport numpy as np
import  numpy as np

from cpython cimport bool


cpdef rebin_array(double [:] arr, int stride, bool mean=False, bool remainder=False):
    """
    rebin arr by a factor stride, using method (ex: np.sum or np.mean), keep the remainder in the
    last bin or not
    """
    cdef int lenb = len(arr) // stride
    cdef double [:] rebinned

    if remainder and len(arr) % stride != 0:
        rebinned = np.empty(lenb + 1)
        if mean: rebinned[-1] = np.mean(arr[lenb*stride:])
        else   : rebinned[-1] = np.sum (arr[lenb*stride:])
    else:
        rebinned = np.empty(lenb)

    cdef int i, s, f
    for i in range(lenb):
        s = i * stride
        f = s + stride
        if mean: rebinned[i] = np.mean(arr[s:f])
        else   : rebinned[i] = np.sum (arr[s:f])

    return np.asarray(rebinned)
