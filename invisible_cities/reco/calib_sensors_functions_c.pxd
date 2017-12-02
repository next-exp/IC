"""
Calibrated response of SiPMs and PMTs
credits: see ic_authors_and_legal.rst in /doc

"""
cimport numpy as np
import numpy as np
#from scipy import signal

"""
Computes the mean of a vector
"""
cpdef double cmean(double [:] v)


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



"""Computes pedestal as average of the waveform"""
cpdef sipm_subtract_baseline_and_normalize(np.ndarray[np.int16_t, ndim=2] sipm,
                                           np.ndarray[np.float64_t, ndim=1] adc_to_pes)


"""Computes pedetal using a MAU"""
cpdef sipm_subtract_baseline_and_normalize_mau(np.ndarray[np.int16_t, ndim=2]sipm,
                                               np.ndarray[np.float64_t, ndim=1] adc_to_pes,
                                               int n_MAU=*)


"""
subtracts the baseline
Uses a MAU to set the signal threshold (thr, in PES)
"""
cpdef sipm_signal_above_thr_mau(np.ndarray[np.int16_t, ndim=2] sipm,
                                np.ndarray[np.float64_t, ndim=1] adc_to_pes,
                                double thr,
                                int n_MAU=*)
