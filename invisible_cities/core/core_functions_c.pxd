cimport numpy as np
import  numpy as np
from cpython cimport bool

cpdef rebin_array(double [:] arr, int stride, bool mean = *, bool remainder = *)
"""
rebin arr by a factor stride, using np.sum or np.mean, keep the remainder in the
last bin or not
"""
