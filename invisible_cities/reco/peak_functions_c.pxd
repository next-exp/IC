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
cpdef _time_from_index(int [:] indx)


"""
Find S1/S2 peaks.
input:
wfzs:   a vector containining the zero supressed wf
indx:   a vector of indexes
returns a dictionary

do not interrupt the peak if next sample comes within stride
accept the peak only if within [l.min, l.max)
accept the peak only if within [t.min, t.max)
"""
cpdef find_S12(double [:] csum, int [:] index,
               time=*, length=*,
               int stride=*, rebin=*, rebin_stride=*)

"""
rebins  a waveform according to stride
The input waveform is a vector such that the index expresses time bin and the
contents expresses energy (e.g, in pes)
The function returns a rebinned vector of T and E.
"""

cpdef correct_S1_ene(S1, np.ndarray csum)

#cpdef rebin_waveform(double [:] t, double[:] e, int stride=*)
cpdef rebin_waveform(int ts, int t_finish, double[:] wf, int stride=*)


"""
subtracts the baseline
Uses a MAU to set the signal threshold (thr, in PES)
returns ZS waveforms for all SiPMs
"""

cpdef signal_sipm(np.ndarray[np.int16_t, ndim=2] SIPM,
                  double [:] adc_to_pes, double thr,
                  int n_MAU=*)

"""
Selects the SiPMs with signal
and returns a dictionary:
input: sipmzs[i,k], where:
       i: runs over number of SiPms (with signal)
       k: runs over waveform.
       sipmzs[i,k] only includes samples above
       threshold (e.g, dark current threshold)
returns {j: [i, sipmzs[i]]}, where:
       j: enumerates sipms with psum >0
       i: sipm ID
"""

cpdef select_sipm(double [:, :] sipmzs)

"""Given a dict with SIPMs (energies above threshold),
return a dict of np arrays, where the key is the sipm
with signal.

input {j: [i, sipmzs[i]]}, where:
       j: enumerates sipms with psum >0
       i: sipm ID
      S2d defining an S2 signal

returns:
      {i, sipm_i[i0:i1]} where:
      i: sipm ID
      sipm_i[i0:i1] waveform corresponding to SiPm i between:
      i0: min index of S2d
      i1: max index of S2d
      only IF the total energy of SiPM is above thr

"""
#cpdef sipm_s2(dict dSIPM, dict S2, double thr)


"""Given a vector with SIPMs (energies above threshold), and a
dictionary of S2s, S2d, returns a dictionary of SiPMs-S2.  Each
index of the dictionary correspond to one S2 and is a list of np
arrays. Each element of the list is the S2 window in the SiPM (if
not zero)

"""
cpdef sipm_s2_dict(dict dSIPM, dict S2d, double thr)
