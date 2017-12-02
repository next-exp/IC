"""
code: calib_sensors_functions_c.pyx
Calibrated response of SiPMs and PMTs
credits: see ic_authors_and_legal.rst in /doc

"""
cimport numpy as np
import  numpy as np
from scipy import signal

from .. core.system_of_units_c import units


cpdef double cmean(double [:] v):
    """Computes the mean of vector v."""
    cdef int  k
    cdef int N = len(v)
    cdef double pmean = 0

    for k in range(N):
        pmean += v[k]
    pmean /= N
    return pmean

cpdef calibrated_pmt_sum(double [:, :]  CWF,
                         double [:]     adc_to_pes,
                         list           pmt_active = [],
                         int            n_MAU = 100,
                         double         thr_MAU =   3):
    """
    Computes the ZS calibrated sum of the PMTs
    after correcting the baseline with a MAU to suppress low frequency noise.
    input:
    CWF:    Corrected waveform (passed by BLR)
    adc_to_pes: a vector with calibration constants
    pmt_active: a list of active PMTs
    n_MAU:  length of the MAU window
    thr_MAU: treshold above MAU to select sample

    """

    cdef int j, k
    cdef int NPMT = CWF.shape[0]
    cdef int NWF  = CWF.shape[1]
    cdef double [:] MAU = np.array(np.ones(n_MAU),
                                   dtype = np.double) * (1 / n_MAU)


    # CWF if above MAU threshold
    cdef double [:, :] pmt_thr  = np.zeros((NPMT,NWF), dtype=np.double)
    cdef double [:]    csum     = np.zeros(      NWF , dtype=np.double)
    cdef double [:]    csum_mau = np.zeros(      NWF , dtype=np.double)
    cdef double [:]    MAU_pmt  = np.zeros(      NWF , dtype=np.double)

    cdef list PMT = list(range(NPMT))
    if len(pmt_active) > 0:
        PMT = pmt_active

    for j in PMT:
        # MAU for each of the PMTs, following the waveform
        MAU_pmt = signal.lfilter(MAU, 1, CWF[j,:])

        for k in range(NWF):
            if CWF[j,k] >= MAU_pmt[k] + thr_MAU: # >= not >: found testing!
                pmt_thr[j,k] = CWF[j,k]

    for j in PMT:
        for k in range(NWF):
            csum_mau[k] += pmt_thr[j, k] * 1 / adc_to_pes[j]
            csum[k] += CWF[j, k] * 1 / adc_to_pes[j]

    return np.asarray(csum), np.asarray(csum_mau)


cpdef calibrated_pmt_mau(double [:, :]  CWF,
                         double [:]     adc_to_pes,
                         list           pmt_active = [],
                         int            n_MAU = 200,
                         double         thr_MAU =   5):
    """
    Returns the calibrated waveforms for PMTs correcting by MAU.
    input:
    CWF:    Corrected waveform (passed by BLR)
    adc_to_pes: a vector with calibration constants
    list: list of active PMTs
    n_MAU:  length of the MAU window
    thr_MAU: treshold above MAU to select sample

    """

    cdef int j, k
    cdef int NPMT = CWF.shape[0]
    cdef int NWF  = CWF.shape[1]
    cdef list PMT = list(range(NPMT))
    if len(pmt_active) > 0:
        PMT = pmt_active


    cdef double [:] MAU = np.array(np.ones(n_MAU),
                                   dtype = np.double) * (1 / n_MAU)

    # CWF if above MAU threshold
    cdef double [:, :] pmt_thr  = np.zeros((NPMT, NWF), dtype=np.double)
    cdef double [:, :] pmt_thr_mau  = np.zeros((NPMT, NWF), dtype=np.double)
    cdef double [:]    MAU_pmt  = np.zeros(      NWF, dtype=np.double)

    for j in PMT:
        # MAU for each of the PMTs, following the waveform
        MAU_pmt = signal.lfilter(MAU, 1, CWF[j,:])

        for k in range(NWF):
            pmt_thr[j,k] = CWF[j,k] * 1 / adc_to_pes[j]
            if CWF[j,k] >= MAU_pmt[k] + thr_MAU: # >= not >: found testing!
                pmt_thr_mau[j,k] = CWF[j,k] * 1 / adc_to_pes[j]

    return np.asarray(pmt_thr), np.asarray(pmt_thr_mau)


cpdef sipm_subtract_baseline_and_normalize(np.ndarray[np.int16_t, ndim=2] sipm,
                                           np.ndarray[np.float64_t, ndim=1] adc_to_pes):
    """Computes pedetal as average of the waveform. Very fast"""
    cdef int NSiPM = sipm.shape[0]
    cdef int NSiWF = sipm.shape[1]
    cdef double [:, :] SiWF = sipm.astype(np.double)
    cdef double [:, :] siwf = np.zeros((NSiPM, NSiWF), dtype=np.double)
    cdef int j, k
    cdef double pmean

    for j in range(NSiPM):
        if adc_to_pes[j] == 0:  # zero calib constant: dead sipm
            continue
        pmean = cmean(SiWF[j])

        for k in range(NSiWF):
            siwf[j,k] = (SiWF[j,k] - pmean) / adc_to_pes[j]

    return np.asarray(siwf)


cpdef sipm_subtract_baseline_and_normalize_mau(np.ndarray[np.int16_t, ndim=2]sipm,
                                               np.ndarray[np.float64_t, ndim=1] adc_to_pes,
                                               int n_MAU=100):
    """Computes pedetal using a MAU. Runs a factor 100 slower than previous"""
    cdef int NSiPM = sipm.shape[0]
    cdef int NSiWF = sipm.shape[1]
    cdef double [:, :] SiWF = sipm.astype(np.double)
    cdef double [:, :] siwf = np.zeros((NSiPM, NSiWF), dtype=np.double)
    cdef double [:] MAU = np.array(np.ones(n_MAU), dtype = np.double) * (1 / n_MAU)
    cdef double [:] MAU_ = np.zeros(NSiWF, dtype=np.double)
    cdef int j, k
    cdef double pmean

    for j in range(NSiPM):
        if adc_to_pes[j] == 0:  # zero calib constant: dead sipm
            continue
        pmean = cmean(SiWF[j])

        for k in range(NSiWF):
            siwf[j,k] = SiWF[j,k] - pmean
        MAU_ = signal.lfilter(MAU, 1, siwf[j,:])

        for k in range(NSiWF):
            siwf[j,k] = (siwf[j,k] - MAU_[k]) / adc_to_pes[j]

    return np.asarray(siwf)

cpdef sipm_signal_above_thr_mau(np.ndarray[np.int16_t, ndim=2] sipm,
                                np.ndarray[np.float64_t, ndim=1] adc_to_pes,
                                thr,
                                int n_MAU=100):
    """
    subtracts the baseline
    Uses a MAU to set the signal threshold (thr, in PES)

    """

    cdef int NSiPM = sipm.shape[0]
    cdef int NSiWF = sipm.shape[1]
    cdef double [:, :] SiWF = sipm.astype(np.double)
    cdef double [:, :] siwf = np.zeros((NSiPM, NSiWF), dtype=np.double)
    cdef double [:] MAU = np.array(np.ones(n_MAU), dtype = np.double) * (1 / n_MAU)
    cdef double [:] MAU_ = np.zeros(NSiWF, dtype=np.double)
    cdef int j, k
    cdef double pmean
    cdef double [:] thrs = np.full(NSiPM, thr)


    # loop over all SiPMs. Skip any SiPM with adc_to_pes constant = 0
    # since this means SiPM is dead
    for j in range(NSiPM):
        if adc_to_pes[j] == 0:
            continue

        # compute and subtract the baseline
        pmean = cmean(SiWF[j])

        for k in range(NSiWF):
            SiWF[j,k] = SiWF[j,k] - pmean
        MAU_ = signal.lfilter(MAU, 1, SiWF[j,:])

        # threshold using the MAU
        for k in range(NSiWF):
            if SiWF[j,k]  > MAU_[k] + thrs[j] * adc_to_pes[j]:
                siwf[j,k] = SiWF[j,k] / adc_to_pes[j]

    return np.asarray(siwf)


cpdef signal_sipm(np.ndarray[np.int16_t, ndim=2] SIPM,
                  double [:] adc_to_pes, thr,
                  int n_MAU=100, int Cal=0):
    """
    subtracts the baseline
    Uses a MAU to set the signal threshold (thr, in PES)
    returns ZS waveforms for all SiPMs

    setting Cal to a non zero value invokes calibration mode
    where the unselected time bins are returned as they are
    instead of set to zero.

    """

    cdef int j, k
    cdef double [:, :] SiWF = SIPM.astype(np.double)
    cdef int NSiPM = SiWF.shape[0]
    cdef int NSiWF = SiWF.shape[1]
    cdef double [:] MAU = np.array(np.ones(n_MAU),
                                   dtype = np.double) * (1 / n_MAU)

    cdef double [:, :] siwf = np.zeros((NSiPM, NSiWF), dtype=np.double)
    cdef double [:]    MAU_ = np.zeros(        NSiWF , dtype=np.double)
    cdef double [:]    thrs = np.full ( NSiPM, thr)
    cdef double pmean

    # loop over all SiPMs. Skip any SiPM with adc_to_pes constant = 0
    # since this means SiPM is dead
    for j in range(NSiPM):
        if adc_to_pes[j] == 0:
            #print('adc_to_pes[{}] = 0, setting sipm waveform to zero'.format(j))
            continue

        # compute and subtract the baseline
        pmean = 0
        for k in range(NSiWF):
            pmean += SiWF[j,k]
        pmean /= NSiWF

        for k in range(NSiWF):
            SiWF[j,k] = SiWF[j,k] - pmean

        # MAU for each of the SiPMs, following the ZS waveform
        MAU_ = signal.lfilter(MAU, 1, SiWF[j,:])

        # threshold using the MAU
        for k in range(NSiWF):
            if Cal != 0:
                if Cal == 1:
                    siwf[j,k] = SiWF[j,k] / adc_to_pes[j]
                else:
                    siwf[j,k] = (SiWF[j,k] - MAU_[k]) / adc_to_pes[j]
            elif SiWF[j,k]  > MAU_[k] + thrs[j] * adc_to_pes[j]:
                siwf[j,k] = SiWF[j,k] / adc_to_pes[j]

    return np.asarray(siwf)
