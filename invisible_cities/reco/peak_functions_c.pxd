"""
Cython version of some peak functions
JJGC December, 2016

"""
cimport numpy as np
import numpy as np
from scipy import signal

"""
computes the ZS calibrated sum of the PMTs
after correcting the baseline with a MAU to suppress low frequency noise.
input:
CWF:    Corrected waveform (passed by BLR)
adc_to_pes: a vector with calibration constants
pmt_active: a list of active PMTs
n_MAU:  length of the MAU window
thr_MAU: treshold above MAU to select sample
"""
cpdef calibrated_pmt_sum(double [:, :] CWF,
                         double [:] adc_to_pes,
                         list       pmt_active = *,
                         int        n_MAU      = *,
                         double thr_MAU        = *)


"""
Return a vector of calibrated PMTs
after correcting the baseline with a MAU to suppress low frequency noise.
input:
CWF:    Corrected waveform (passed by BLR)
adc_to_pes: a vector with calibration constants
pmt_active: a list of active PMTs
n_MAU:  length of the MAU window
thr_MAU: treshold above MAU to select sample
"""
cpdef calibrated_pmt_mau(double [:, :]  CWF,
                         double [:] adc_to_pes,
                         list       pmt_active = *,
                         int        n_MAU      = *,
                         double     thr_MAU    = *)


"""
takes a waveform wf and returns the values of the wf above threshold:
if the input waveform is of the form [e1,e2,...en],
where ei is the energy of sample i,
then then the algorithm returns a vector [e1,e2...ek],
where k <=n and ei > threshold and
a vector of indexes [i1,i2...ik] which label the position
of the zswf of [e1,e2...ek]
For example if the input waveform is:
[1,2,3,5,7,8,9,9,10,9,8,5,7,5,6,4,1] and the trhesold is 5
then the algoritm returns
a vector of amplitudes [7,8,9,9,10,9,8,7,6] and a vector of indexes
[4,5,6,7,8,9,10,12,14]
"""
cpdef wfzs(double [:] wf, double threshold=*)


"""
returns the times (in ns) corresponding to the indexes in indx
"""
cpdef time_from_index(int [:] indx)


"""
Find S1/S2 peaks.
input:
wfzs:   a vector containining the zero supressed wf
indx:   a vector of indexes
returns a dictionary

do not interrupt the peak if next sample comes within stride
accept the peak only if within [lmin, lmax)
accept the peak only if within [tmin, tmax)
"""
cpdef find_S12(double [:] wfzs, int [:] index,
               double tmin=*, double tmax=*,
               int lmin=*, int lmax=*,
               int stride=*, rebin=*, rebin_stride=*)

"""
rebins  a waveform according to stride
The input waveform is a vector such that the index expresses time bin and the
contents expresses energy (e.g, in pes)
The function returns a rebinned vector of T and E.
"""

cpdef correct_S1_ene(S1, np.ndarray csum)

cpdef rebin_waveform(double [:] t, double[:] e, int stride=*)


"""
subtracts the baseline
Uses a MAU to set the signal threshold (thr, in PES)
returns ZS waveforms for all SiPMs
"""

cpdef rebin_S2(double [:] t, double [:] e, dict sipms, int nrebin)

cpdef signal_sipm(np.ndarray[np.int16_t, ndim=2] SIPM,
                  double [:] adc_to_pes, double thr,
                  int n_MAU=*)

"""
Selects the SiPMs with signal
and returns a dictionary
"""

cpdef select_sipm(double [:, :] sipmzs)
