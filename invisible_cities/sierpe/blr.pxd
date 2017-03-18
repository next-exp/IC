"""
Definition file for BLR algorithm
JJGC December 2016

deconvolve_signal takes the raw signal output from a PMT FEE
and returns the deconvolved (BLR) waveform.
input:
signal_daq: the raw signal (in doubles)
n_baseline: length of the baseline to compute the mean. It tipically takes
            most of the baseline, since the mean over the whole raw signal is
            zero (thus the calculation including the full window gives
                  the pedestal accurately)
coeff_clean: the coefficient for the cleaning part of the BLR algorithm
coef_blr: the coefficient for the accumulator part of the BLR algorithm
thr_trigger: threshold to decide whether signal is on
acum_discharge_length: length to dicharge the accumulator in the absence
                       of signal.


deconv_pmt performs the deconvolution for the PMTs of the energy plane
input:
pmtrwf:    the raw waveform for all the PMTs (shorts)
coeff_c:   a vector of deconvolution coefficients (cleaning)
coeff_blr: a vector of deconvolution coefficients (blr)
n_baseline, thr_trigger as described above

"""
import numpy as np
cimport numpy as np

cpdef deconvolve_signal(double [:] signal_daq,
                        int        n_baseline     = *,
                        double     coef_clean     = *,
                        double     coef_blr       = *,
                        double     thr_trigger    = *,
                        int acum_discharge_length = *)

cpdef deconv_pmt(np.ndarray[np.int16_t, ndim=2] pmtrwf,
                 double [:]                     coeff_c,
                 double [:]                     coeff_blr,
                 list                           pmt_active  = *,
                 int                            n_baseline  = *,
                 double                         thr_trigger = *)
