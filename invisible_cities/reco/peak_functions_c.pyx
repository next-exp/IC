"""
Functions used for peak finding.
Last revision: June 2017, JJGC

**Public functions**


calibrated_pmt_sum(double [:, :]  CWF,
                         double [:]     adc_to_pes,
                         list           pmt_active = [],
                         int            n_MAU = 100,
                         double         thr_MAU =   3)

  Compute the ZS calibrated sum of the PMTs
  after correcting the baseline with a MAU to suppress low frequency noise.

calibrated_pmt_mau(double [:, :]  CWF,
                           double [:]     adc_to_pes,
                           list           pmt_active = [],
                           int            n_MAU = 200,
                           double         thr_MAU =   5):

   Returns the calibrated waveforms for PMTs correcting by MAU.

wfzs(double [:] wf, double threshold=0)
  Takes a waveform wf and return the values of the wf above threshold:

rebin_waveform(double [:] t, double[:] e, int stride = 40)
  Rebin a waveform according to stride

cpdef correct_S1_ene(S1, np.ndarray csum)??

find_S12(double [:] wfzs,  int [:] index,
               time=(), length=(),
               int stride=4, rebin=False, rebin_stride=40)
  Find S1/S2 peaks. Wrapper around the cython version returning instances
  of S12 class.


signal_sipm(np.ndarray[np.int16_t, ndim=2] SIPM,
                    double [:] adc_to_pes, double thr,
                    int n_MAU=100):

   subtracts the baseline
   Uses a MAU to set the signal threshold (thr, in PES)
   returns ZS waveforms for all SiPMs

cpdef select_sipm(double [:, :] sipmzs)
       Selects the SiPMs with signal
       and returns a dictionary

** Private **

cpdef _time_from_index(int [:] indx):
  returns the times (in ns) corresponding to the indexes in indx



"""
cimport numpy as np
import  numpy as np
from scipy import signal

from .. evm.pmaps import S1
from .. evm.pmaps import S2
from .. evm.pmaps import S2Si

from .. core.system_of_units_c import units

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
    cdef double [:, :] pmt_thr  = np.zeros((NPMT,NWF), dtype=np.double)
    cdef double [:, :] pmt_thr_mau  = np.zeros((NPMT,NWF), dtype=np.double)
    cdef double [:]    MAU_pmt  = np.zeros(      NWF , dtype=np.double)

    for j in PMT:
        # MAU for each of the PMTs, following the waveform
        MAU_pmt = signal.lfilter(MAU, 1, CWF[j,:])

        for k in range(NWF):
            pmt_thr[j,k] = CWF[j,k] * 1 / adc_to_pes[j]
            if CWF[j,k] >= MAU_pmt[k] + thr_MAU: # >= not >: found testing!
                pmt_thr_mau[j,k] = CWF[j,k] * 1 / adc_to_pes[j]


    return np.asarray(pmt_thr), np.asarray(pmt_thr_mau)


cpdef wfzs(double [:] wf, double threshold=0):
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
    cdef int len_wf = wf.shape[0]
    cdef double [:] wfzs_e = np.zeros(len_wf, dtype=np.double)
    cdef int    [:] wfzs_i = np.zeros(len_wf, dtype=np.int32)

    cdef int i,j
    j = 0
    for i in range(len_wf):
        if wf[i] > threshold:
            wfzs_e[j] = wf[i]
            wfzs_i[j] =    i
            j += 1

    cdef double [:] wfzs_ene  = np.zeros(j, dtype=np.double)
    cdef int    [:] wfzs_indx = np.zeros(j, dtype=np.int32)

    for i in range(j):
        wfzs_ene [i] = wfzs_e[i]
        wfzs_indx[i] = wfzs_i[i]

    return np.asarray(wfzs_ene), np.asarray(wfzs_indx)


cpdef _time_from_index(int [:] indx):
    """
    returns the times (in ns) corresponding to the indexes in indx
    """
    cdef int len_indx = indx.shape[0]
    cdef double [:] tzs = np.zeros(len_indx, dtype=np.double)

    cdef int i
    cdef double step = 25 #ns
    for i in range(len_indx):
        tzs[i] = step * float(indx[i])

    return np.asarray(tzs)


cpdef find_s1(double [:] csum,  int [:] index,
              time, length,
              int stride=4, rebin=False, rebin_stride=4):
    """
    find s1 peaks and returns S1 objects
    """

    return S1(find_s12(csum, index, time, length, stride, rebin, rebin_stride))


cpdef find_s2(double [:] csum,  int [:] index,
              time, length,
              int stride=40, rebin=True, rebin_stride=40):
    """
    find s2 peaks and returns S2 objects
    """

    return S2(find_s12(csum, index, time, length, stride, rebin, rebin_stride))


cpdef find_s2si(double [:, :] sipmzs, dict s2d, double thr):
    """
    find s2si and returns S2Si objects
    """
    s2sid = sipm_s2sid(sipmzs, s2d, thr)
    return S2Si(s2d, s2sid)


cpdef find_s12(double [:] csum,  int [:] index,
               time=(), length=(),
               int stride=4, rebin=False, rebin_stride=40):
    """
    Find S1/S2 peaks.
    input:
    csum:   a summed (across pmts) waveform
    indx:   a vector of indexes
    returns a dictionary
    do not interrupt the peak if next sample comes within stride
    accept the peak only if within [lmin, lmax)
    accept the peak only if within [tmin, tmax)
    returns a dictionary of S12
    """
    cdef double tmin, tmax
    cdef int    limn, lmax
    #cdef double [:] P = wfzs
    cdef double [:] T = _time_from_index(index)
    cdef dict S12  = {}
    cdef dict S12L = {}
    cdef int i, j, k, ls, i_i, i_min
    tmin, tmax = time
    lmin, lmax = length

    i_min = tmin / (25*units.ns)
    i_i = np.where(np.asarray(index) >= i_min)[0].min() #first index of index with t > tmin
    S12[0] = np.array([index[i_i], index[i_i] + 1], dtype=np.int32)
    j = 0
    for i in range(i_i + 1, len(index)):
        assert T[i] > tmin
        if T[i] > tmax: break
        # New s12, create new start and end index
        elif index[i] - stride > index[i-1]:
            j += 1
            S12[j] = np.array([index[i], index[i] + 1], dtype=np.int32)
        # Update end index in current S12
        else: S12[j][1] = index[i] + 1

    # re-arrange and rebin
    j = 0
    for i_peak in S12.values():
        if not (lmin <= i_peak[1] - i_peak[0] < lmax):
            continue
        S12wf = csum[i_peak[0]: i_peak[1]]
        if rebin == True:
            TR, ER = rebin_waveform(*_time_from_index(i_peak), S12wf, stride=rebin_stride)
            S12L[j] = [TR, ER]
        else:
            S12L[j] = [np.arange(*_time_from_index(i_peak), 25*units.ns), np.asarray(S12wf)]
        j += 1

    return S12L


# cpdef correct_S1_ene(S1, np.ndarray csum):
#     cdef dict S1_corr = {}
#     for peak_no, (t, _) in S1.items():
#         indices          = (t // 25).astype(int)
#         S1_corr[peak_no] = t, csum[indices]
#     return S12(S1_corr)


cpdef correct_s1_ene(dict s1d, np.ndarray csum):
    cdef dict S1_corr = {}
    cdef int peak_no
    for peak_no, (t, _) in s1d.items():
        indices          = (t // 25).astype(int)
        S1_corr[peak_no] = t, csum[indices]
    return S1(S1_corr)


cpdef rebin_waveform(int ts, int t_finish, double[:] wf, int stride=40):
    """
    Rebin a waveform according to stride
    The input waveform is a vector such that the index expresses time bin and the
    contents expresses energy (e.g, in pes)
    The function returns the rebinned T& E vectors.
    """

    assert (ts < t_finish)
    cdef int  bs = 25*units.ns # bin size
    cdef int rbs = bs * stride # rebinned bin size

    # Find the nearest time (in stride samples) before ts
    cdef int t_start  = (ts // (rbs)) * rbs
    cdef int t_total  = t_finish - t_start
    cdef int n = t_total // (rbs)
    cdef int r = t_total  % (rbs)

    lenb = n
    if r > 0: lenb = n+1

    cdef double [:] T = np.zeros(lenb, dtype=np.double)
    cdef double [:] E = np.zeros(lenb, dtype=np.double)

    cdef int j = 0
    cdef int i, tb
    cdef double esum
    for i in range(n):
        esum = 0
        for tb in range(int(t_start +     i*rbs),
                        int(t_start + (1+i)*rbs),
                        int(bs)):
            if tb < ts: continue
            esum  += wf[j]
            j     += 1

        E[i] = esum
        if i == 0: T[i] = np.mean((ts, t_start + rbs))
        else     : T[i] = t_start + i*rbs + rbs/2.0

    if r > 0:
        esum  = 0
        for tb in range(int(t_start + n*rbs),
                       int(t_finish),
                       int(bs)):
            if tb < ts:continue
            esum  += wf[j]
            j     += 1

        E[n] = esum
        if n == 0: T[n] = np.mean((ts, t_finish))
        else     : T[n] = (t_start + n*rbs + t_finish) / 2.0

    assert j == len(wf)
    return np.asarray(T), np.asarray(E)


cpdef signal_sipm(np.ndarray[np.int16_t, ndim=2] SIPM,
                  double [:] adc_to_pes, double thr,
                  int n_MAU=100):
    """
    subtracts the baseline
    Uses a MAU to set the signal threshold (thr, in PES)
    returns ZS waveforms for all SiPMs

    """

    cdef int j, k
    cdef double [:, :] SiWF = SIPM.astype(np.double)
    cdef int NSiPM = SiWF.shape[0]
    cdef int NSiWF = SiWF.shape[1]
    cdef double [:] MAU = np.array(np.ones(n_MAU),
                                   dtype = np.double) * (1 / n_MAU)

    cdef double [:, :] siwf = np.zeros((NSiPM, NSiWF), dtype=np.double)
    cdef double [:]    MAU_ = np.zeros(        NSiWF , dtype=np.double)
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
            if SiWF[j,k]  > MAU_[k] + thr * adc_to_pes[j]:
                siwf[j,k] = SiWF[j,k] / adc_to_pes[j]

    return np.asarray(siwf)


cpdef select_sipm(double [:, :] sipmzs):
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
    cdef int NSIPM = sipmzs.shape[0]
    cdef int NWFM = sipmzs.shape[1]
    cdef dict SIPM = {}
    cdef int i, j, k
    cdef double psum

    j = 0
    for i in range(NSIPM):
        psum = 0
        for k in range(NWFM):
            psum += sipmzs[i,k]
        if psum > 0:
            SIPM[j] = [i,np.asarray(sipmzs[i])]
            j += 1
    return SIPM


cdef _index_from_s2(list s2l):
    """Return the indexes defining the vector."""
    cdef int t0 = int(s2l[0][0] // units.mus)
    return t0, t0 + len(s2l[0])



cdef sipm_s2(dict dSIPM, list s2l, double thr):
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

    cdef int i0, i1, ID
    i0, i1 = _index_from_s2(s2l)
    cdef dict SIPML = {}
    cdef double psum

    for ID, sipm in dSIPM.values():
        slices = sipm[i0:i1]
        psum = np.sum(slices)
        if psum > thr:
            SIPML[ID] = slices.astype(np.double)
    return SIPML


cdef sipm_s2sid(double [:, :] sipmzs, dict s2d, double thr):
    """Given a vector with SIPMs (energies above threshold), and a
    dictionary S2d, returns a dictionary of SiPMs-S2.  Each
    index of the dictionary correspond to one S2 and is a list of np
    arrays. Each element of the list is the S2 window in the SiPM (if
    not zero)

    """
    # dict dSIPM
    # select_sipm(sipmzs)
    cdef int peak_no
    cdef list s2l
    cdef dict s2si = {}
    for peak_no, s2l in s2d.items():
        s2si[peak_no] = sipm_s2(select_sipm(sipmzs), s2l, thr=thr)

    return s2si
