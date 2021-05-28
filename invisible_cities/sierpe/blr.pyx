#cython: language_level=3
import  numpy as np
cimport numpy as np
from scipy import signal as SGN

cpdef deconvolve_signal(double [:] signal_daq,
                        double coeff_clean            = 2.905447E-06,
                        double coeff_blr              = 1.632411E-03,
                        double thr_trigger            =     5,
                        int    accum_discharge_length =  5000):

    """
    The accumulator approach by Master VHB
    decorated and cythonized  by JJGC
    Current version using memory views

    In this version the recovered signal and the accumulator are
    always being charged. At the same time, the accumulator is being
    discharged when there is no signal. This avoids runoffs
    """

    cdef double coef = coeff_blr
    cdef double thr_acum = thr_trigger / coef
    cdef int len_signal_daq = len(signal_daq)

    cdef double [:] signal_r = np.zeros(len_signal_daq, dtype=np.double)
    cdef double [:] acum     = np.zeros(len_signal_daq, dtype=np.double)

    cdef int j

    # compute noise
    cdef double noise =  0
    cdef int nn = 400 # fixed at 10 mus

    for j in range(nn):
        noise += signal_daq[j] * signal_daq[j]
    noise /= nn
    cdef double noise_rms = np.sqrt(noise)

    # trigger line
    cdef double trigger_line = thr_trigger * noise_rms

    # cleaning signal
    cdef double [:]  b_cf
    cdef double [:]  a_cf

    b_cf, a_cf = SGN.butter(1, coeff_clean, 'high', analog=False);
    signal_daq = SGN.lfilter(b_cf, a_cf, signal_daq)

    cdef int k
    j = 0
    signal_r[0] = signal_daq[0]
    for k in range(1, len_signal_daq):

        # always update signal and accumulator
        signal_r[k] = (signal_daq[k] + signal_daq[k]*(coef / 2) +
                       coef * acum[k-1])

        acum[k] = acum[k-1] + signal_daq[k]

        if (signal_daq[k] < trigger_line) and (acum[k-1] < thr_acum):
            # discharge accumulator
            if acum[k-1] > 1:
                acum[k] = acum[k-1] * (1 - coef)
                if j < accum_discharge_length - 1:
                    j = j + 1
                else:
                    j = accum_discharge_length - 1
            else:
                acum[k] = 0
                j = 0
    # return recovered signal
    return np.asarray(signal_r)


cpdef deconvolve_signal_fpga( short [:] signal_daq
                            , double    coeff_clean = 2.905447E-06
                            , double    coeff_blr   = 1.632411E-03
                            , double    thr_trigger = 5
                            , size_t    base_window = 1024        ):

    """
    Simulate the deconvolution process as in the daq, differences compared to
    usual offline deconvolution:
      - Baseline is calculated as a moving average of 1024 counts (FPGA).
      - Flips result between raw and deconvolved signal when outside of both
        signal and discharge regions.
      - Discharge made at fixed value (0.995)
      - Result is truncated to integers.
      - ADC threshold acting as absolute threshold.

    Parameters
    ----------
    signal_daq  : short array
         PMT raw waveform
    coeff_clean : double
         Characteristic parameter of the high pass filter
    coeff_blr   : double
         Characteristic parameter of BLR
    thr_trigger : double
         Threshold in ADCs to activate BLR
    base_window : size
         Moving average window for baseline calculation

    Returns
    ----------
    Tuple of arrays:
    - Deconvolved waveform (int16  [:])
    - Baseline             (int16  [:])
    """
    cdef double coef = coeff_blr
    cdef double thr_acum = thr_trigger / coef
    cdef int len_signal_daq = len(signal_daq)

    cdef double [:] signal_r = np.zeros(len_signal_daq, dtype=np.double)
    cdef double [:] signal_f = np.zeros(len_signal_daq, dtype=np.double)
    cdef double [:] acum     = np.zeros(len_signal_daq, dtype=np.double)

    cdef int j

    # compute noise
    cdef double noise =  0
    cdef int nn = 400 # fixed at 10 mus

    for j in range(nn):
        noise += signal_daq[j] * signal_daq[j]
    noise /= nn
    cdef double noise_rms = np.sqrt(noise)

    # trigger line
    cdef double trigger_line = thr_trigger #* noise_rms
    # cleaning signal
    cdef double [:]  b_cf
    cdef double [:]  a_cf

    ### Baseline related variables
    cdef double [:]  top   = np.zeros(len_signal_daq, dtype=np.double)
    cdef short  [:]  aux   = np.zeros(len_signal_daq, dtype=np.int16)
    cdef long ped          = np.sum  (signal_daq[0:base_window], dtype=np.int32)
    cdef size_t delay      = 2 # Delay in the FPGA in the bin used for baseline substraction
    cdef unsigned int iaux = base_window

    top[0:base_window] = ped/base_window
    aux[0:base_window] = signal_daq[0:base_window]

    b_cf, a_cf = SGN.butter(1, coeff_clean, 'high', analog=False);
    g, a1 = b_cf[0], -a_cf[-1]

    ### Initiate filt
    filt_hr   = 0
    #1st
    filt_x    = g  * -(signal_daq[0] - top[0])
    filt_a1hr = a1 * filt_hr

    filt_h    = filt_x + filt_a1hr
    signal_f[0] = filt_h - filt_hr
    filt_hr     = filt_h

    #2nd
    filt_x      = g  * -(signal_daq[1] - top[0])
    filt_a1hr   = a1 * filt_hr

    filt_h      = filt_x + filt_a1hr
    signal_f[1] = filt_h - filt_hr
    filt_hr     = filt_h


    cdef int k

    #Initiate BLR
    for k in range(0, delay):
        signal_r[k] = signal_daq[k]

    for k in range(2, len_signal_daq):
        # always update signal and accumulator
        current     = signal_daq[k]
        baseline    = top[k-delay]

        ### High-pass filter
        filt_x      = g  * -(current - baseline)
        filt_a1hr   = a1 * filt_hr

        filt_h      = filt_x + filt_a1hr
        signal_f[k] = filt_h - filt_hr

        ### BLR restoration
        blr_val     = (signal_f[k] + signal_f[k]*(coef / 2) +
                       coef * acum[k-1]) + baseline

        acum[k] = acum[k-1] + signal_f[k]

        if (signal_f[k] < trigger_line) and (acum[k-1] < thr_acum):
            # discharge accumulator
            if acum[k-1] > 1:
                acum[k] = acum[k-1] * .995 #Fixed discharge in FPGA code
            else:
                acum [k] = 0
                blr_val = current # When outside of BLR/discharge, flip to raw signal
                if k>=base_window:
                    ped       = ped + current - aux[iaux-base_window]
                    aux[iaux] = current
                    iaux     += 1

        signal_r[k] = blr_val
        top[k]      = ped/base_window

    return np.asarray(signal_r, dtype='int16'), np.asarray(top, dtype='int16')
