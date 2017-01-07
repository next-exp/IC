
"""
Performance of some coreFunctions
"""

from time import time
import numpy as np
import invisible_cities.core.wfm_functions as wfm
import invisible_cities.cython.core.peak_functions as cpf


t = np.arange(1.,100., 0.1, dtype=np.double)
e = np.exp(-t/t**2)

t0 = time()
T, E = wfm.rebin_wf(t, e, stride=2)
t1 = time()
dt = t1 - t0
print("rebin_wf (wfmFunctions) run in {} s".format(dt))

t0 = time()
T, E = wfm.rebin_waveform(t, e, stride=2)
t1 = time()
dt = t1 - t0
print("rebin_waveform (wfmFunctions) run in {} s".format(dt))

t0 = time()
T, E = cpf.rebin_waveform(t, e, stride=2)
t1 = time()
dt = t1 - t0
print("cython rebin_waveform (ICython/Core/peakFunctions) run in {} s".format(dt))
