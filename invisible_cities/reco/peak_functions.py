"""Functions to find peaks, S12 selection etc.
Last revision: JJGC, June, 2017


The list of the functions is:

**Private functions used for testing**
(the corresponding public functions without the _  in fron of the name are
in module peak_functions_c)

_calibrated_pmt_sum(CWF, adc_to_pes, pmt_active = [], n_MAU=200, thr_MAU=5)
  Compute the ZS calibrated sum of the PMTs
  after correcting the baseline with a MAU to suppress low frequency noise.

_wfzs(wf, threshold=0)
  Takes a waveform wf and return the values of the wf above threshold:

_time_from_index(indx)
  Return the times (in ns) corresponding to the indexes in indx

_rebin_waveform(t, e, stride=40)
  Rebin a waveform according to stride

_find_S12(wfzs, index,
          time   = minmax(0, 1e+6),
          length = minmax(8, 1000000),
          stride=4, rebin=False, rebin_stride=40)
  Find S1/S2 peaks.

**Public functions**

find_S12(wfzs, index,
             time   = minmax(0, 1e+6),
             length = minmax(8, 1000000),
             stride=4, rebin=False, rebin_stride=40)
  Find S1/S2 peaks. Wrapper around the cython version returning instances
  of S12 class.

sipm_s2_dict(SIPM, S2d, thr=5 * units.pes)
  Given a vector with SIPMs (energies above threshold), and a
  dictionary of S2s, S2d, returns a dictionary of SiPMs-S2.

sipm_s2(dSIPM, S2, thr=5*units.pes)
  Given a vector with SIPMs (energies above threshold), return a dict
  of np arrays, where the key is the sipm with signal.

compute_csum_and_pmaps(event,pmtrwf, sipmrwf, s1par, s2par, thresholds,
                       calib_vectors, deconv_params)
    Compute calibrated sum and PMAPS.

"""


import numpy  as np

from scipy import signal

from .. core   import system_of_units as units
from .. sierpe import blr

from .. io import pmap_io as pio
from .         import peak_functions_c as cpf
from .  params import CalibratedSum
from .  params import PMaps
from .. core.ic_types          import minmax

def _calibrated_pmt_sum(CWF, adc_to_pes, pmt_active = [], n_MAU=200, thr_MAU=5):
    """Compute the ZS calibrated sum of the PMTs
    after correcting the baseline with a MAU to suppress low frequency noise.
    input:
    CWF         : Corrected waveform (passed by BLR)
    adc_to_pes  : a vector with calibration constants
    n_MAU       : length of the MAU window
    thr_MAU     : treshold above MAU to select sample

    NB: This function is used only for testing purposes, thus the
    annotation _name marking it as private (e.g, not in the API). It is
    programmed "c-style", which is not necesarily optimal in python,
    but follows the same logic that the corresponding cython function
    (in peak_functions_c), which runs faster and should be used
    instead of this one for nornal calculations.
    """

    NPMT = CWF.shape[0]
    NWF  = CWF.shape[1]
    MAU  = np.array(np.ones(n_MAU), dtype=np.double) * (1 / n_MAU)

    pmt_thr  = np.zeros((NPMT, NWF), dtype=np.double)
    csum     = np.zeros(       NWF,  dtype=np.double)
    csum_mau = np.zeros(       NWF,  dtype=np.double)
    MAU_pmt  = np.zeros(       NWF,  dtype=np.double)

    MAUL = []
    PMT = list(range(NPMT))
    if len(pmt_active) > 0:
        PMT = pmt_active

    for j in PMT:
        # MAU for each of the PMTs, following the waveform
        MAU_pmt = signal.lfilter(MAU, 1, CWF[j,:])
        MAUL.append(MAU_pmt)
        csum += CWF[j] * 1 / adc_to_pes[j]
        for k in range(NWF):
            if CWF[j,k] >= MAU_pmt[k] + thr_MAU: # >= not >. Found testing
                pmt_thr[j,k] = CWF[j,k]
        csum_mau += pmt_thr[j] * 1 / adc_to_pes[j]
    return csum, csum_mau, np.array(MAUL)


def _wfzs(wf, threshold=0):
    """Takes a waveform wf and return the values of the wf above
    threshold: if the input waveform is of the form [e1,e2,...en],
    where ei is the energy of sample i, then then the algorithm
    returns a vector [e1,e2...ek], where k <=n and ei > threshold and
    a vector of indexes [i1,i2...ik] which label the position of the
    zswf of [e1,e2...ek]

    For example if the input waveform is:
    [1,2,3,5,7,8,9,9,10,9,8,5,7,5,6,4,1] and the trhesold is 5
    then the algoritm returns
    a vector of amplitudes [7,8,9,9,10,9,8,7,6] and a vector of indexes
    [4,5,6,7,8,9,10,12,14]

    NB: This function is used only for testing purposed (thus private). It is
    programmed "c-style", which is not necesarily optimal in python,
    but follows the same logic that the corresponding cython function
    (in peak_functions_c), which runs faster and should be used
    instead of this one for nornal calculations.
    """
    len_wf = wf.shape[0]
    wfzs_e = np.zeros(len_wf, dtype=np.double)
    wfzs_i = np.zeros(len_wf, dtype=np.int32)
    j=0
    for i in range(len_wf):
        if wf[i] > threshold:
            wfzs_e[j] = wf[i]
            wfzs_i[j] =    i
            j += 1

    wfzs_ene  = np.zeros(j, dtype=np.double)
    wfzs_indx = np.zeros(j, dtype=np.int32)

    for i in range(j):
        wfzs_ene [i] = wfzs_e[i]
        wfzs_indx[i] = wfzs_i[i]

    return wfzs_ene, wfzs_indx


def _time_from_index(indx):
    """Return the times (in ns) corresponding to the indexes in indx

    NB: This function is used mainly for testing purposed (private). It is
    programmed "c-style", which is not necesarily optimal in python,
    but follows the same logic that the corresponding cython function
    (in peak_functions_c), which runs faster and should be used
    instead of this one for nornal calculations.
    """
    len_indx = indx.shape[0]
    tzs = np.zeros(len_indx, dtype=np.double)

    step = 25 #ns
    for i in range(len_indx):
        tzs[i] = step * float(indx[i])

    return tzs


def _rebin_waveform(ts, t_finish, wf, stride=40):
    """
    Rebin a waveform according to stride
    The input waveform is a vector such that the index expresses time bin and the
    contents expresses energy (e.g, in pes)
    The function returns the rebinned T and E vectors

    NB: This function is used mainly for testing purposes (private). It is
     programmed "c-style", which is not necesarily optimal
    in python, but follows the same logic that the corresponding cython
    function (in peak_functions_c), which runs faster and should be used
    instead of this one for nornal calculations.

    Parameters:
    t_s:       starting time for waveform
    t_finish:  end time for waveform
    wf:        wavform (chunk)
    stride:    How many (25 ns) samples we combine into a single bin
    """

    assert (ts < t_finish)

    # Find the nearest time (in stride samples) before ts
    t_start  = int((ts // (stride*25*units.ns)) * stride*25*units.ns)
    t_total  = t_finish - t_start
    n = int(t_total // (stride*25*units.ns))  # number of samples
    r = int(t_total  % (stride*25*units.ns))

    lenb = n
    if r > 0: lenb = n+1

    T = np.zeros(lenb, dtype=np.double)
    E = np.zeros(lenb, dtype=np.double)

    j = 0
    for i in range(n):
        esum  = 0
        for tb in range(int(t_start +  i   *stride*25*units.ns),
                        int(t_start + (i+1)*stride*25*units.ns),
                        int(25*units.ns)):
            if tb < ts: continue
            esum += wf[j]
            j    += 1

        E[i] = esum
        if i == 0: T[0] = (ts + t_start +   stride*25*units.ns) / 2
        else     : T[i] = (     t_start + i*stride*25*units.ns + stride*25*units.ns/2)

    if r > 0:
        esum  = 0
        for tb in range(int(t_start + n*stride*25*units.ns),
                        int(t_finish),
                        int(25*units.ns)):
            if tb < ts: continue
            esum += wf[j]
            j    += 1

        E[n] = esum
        if n == 0: T[n] = (ts + t_finish) / 2
        else     : T[n] = (t_start + n*stride*25*units.ns + t_finish) / 2

    assert j == len(wf) # ensures you have rebinned correctly the waveform
    return T, E


def find_S12(csum, index,
              time   = minmax(0, 1e+6),
              length = minmax(8, 1000000),
              stride=4, rebin=False, rebin_stride=40):
    """
    Find S1/S2 peaks.
    input:
    wfzs:   a vector containining the zero supressed wf
    indx:   a vector of indexes
    returns a dictionary

    do not interrupt the peak if next sample comes within stride
    accept the peak only if within [lmin, lmax)
    accept the peak only if within [tmin, tmax)
    returns a dictionary of S12

    NB: This function is used mainly for testing purposed. It is programmed
     "c-style", which is not necesarily optimal
    in python, but follows the same logic that the corresponding cython
    function (in peak_functions_c), which runs faster and should be used
    instead of this one for nornal calculations.
    """

    T = cpf._time_from_index(index)

    S12  = {}
    S12L = {}

    # Start end end index of S12, [start i, end i)
    S12[0] = np.array([index[0], index[0] + 1], dtype=np.int32)

    j = 0
    for i in range(1, len(index)) :

        if T[i] > time.max: break
        if T[i] < time.min: continue

        # New s12, create new start and end index
        if index[i] - stride > index[i-1]:
            j += 1
            S12[j] = np.array([index[i], index[i] + 1], dtype=np.int32)

        # Update end index in current S12
        S12[j][1] = index[i] + 1

    j = 0
    for i_peak in S12.values():

        if not (length.min <= i_peak[1] - i_peak[0] < length.max):
            continue

        S12wf = csum[i_peak[0]: i_peak[1]]
        if rebin == True:
            TR, ER = _rebin_waveform(*cpf._time_from_index(i_peak), S12wf, stride=rebin_stride)
            S12L[j] = [TR, ER]
        else:
            S12L[j] = [np.arange(*cpf._time_from_index(i_peak), 25*units.ns), S12wf]
        j += 1

    return pio.S12(S12L)


# def find_S12(wfzs, index,
#              time   = minmax(0, 1e+6),
#              length = minmax(8, 1000000),
#              stride=4, rebin=False, rebin_stride=40):
#     """
#     Find S1/S2 peaks.
#     input:
#     wfzs:   a vector containining the zero supressed wf
#     indx:   a vector of indexes
#     returns a dictionary
#
#     do not interrupt the peak if next sample comes within stride
#     accept the peak only if within [lmin, lmax)
#     accept the peak only if within [tmin, tmax)
#     returns a dictionary of S12
#
#     NB: This function is a wrapper around the cython function. It returns
#     a dictionary of namedtuples (Waveform(t = [t], E = [E])), where
#     [t] and [E] are np arrays.
#     """
#
#     from collections import namedtuple
#
#     Waveform = namedtuple('Waveform', 't E')
#
#     S12 = cpf.find_S12(wfzs, index,
#                        *t, *l,
#                       stride,
#                       rebin, rebin_stride)
#
#     return {i: Waveform(t, E) for i, (t,E) in S12.items()}

def sipm_s2_dict(SIPM, S2d, thr=5 * units.pes):
    """Given a vector with SIPMs (energies above threshold), and a
    dictionary of S2s, S2d, returns a dictionary of SiPMs-S2.  Each
    index of the dictionary correspond to one S2 and is a list of np
    arrays. Each element of the list is the S2 window in the SiPM (if
    not zero)
    """
    return {i: sipm_s2(SIPM, S2, thr=thr) for i, S2 in S2d.items()}


def sipm_s2(dSIPM, S2, thr=5*units.pes):
    """Given a vector with SIPMs (energies above threshold), return a dict
    of np arrays, where the key is the sipm with signal.
    """

    def index_from_s2(S2):
        """Return the indexes defining the vector."""
        t0 = int(S2[0][0] // units.mus)
        return t0, t0 + len(S2[0])

    i0, i1 = index_from_s2(S2)
    SIPML = {}
    for ID, sipm in dSIPM.values():
        slices = sipm[i0:i1]
        psum = np.sum(slices)
        if psum > thr:
            SIPML[ID] = slices.astype(np.double)
    return SIPML


def compute_csum_and_pmaps(event, pmtrwf, sipmrwf, s1par, s2par, thresholds,
                        calib_vectors, deconv_params):
    """Compute calibrated sum and PMAPS.

    :param pmtrwf: PMTs RWF
    :param sipmrwf: SiPMs RWF
    :param s1par: parameters for S1 search (S12Params namedtuple)
    :param s2par: parameters for S2 search (S12Params namedtuple)
    :param thresholds: thresholds for searches (ThresholdParams namedtuple)
                       ('ThresholdParams',
                        'thr_s1 thr_s2 thr_MAU thr_sipm thr_SIPM')
    :param pmt_active: a list specifying the active (not dead) pmts
                       in the event. An empty list implies all active.
    :param n_baseline:  number of samples taken to compute baseline
    :param thr_trigger: threshold to start the BLR process
    :param event: event number

    :returns: a nametuple of calibrated sum and a namedtuple of PMAPS
    """
    s1_params = s1par
    s2_params = s2par
    thr = thresholds

    adc_to_pes = calib_vectors.adc_to_pes
    coeff_c    = calib_vectors.coeff_c
    coeff_blr  = calib_vectors.coeff_blr
    adc_to_pes_sipm = calib_vectors.adc_to_pes_sipm
    pmt_active = calib_vectors.pmt_active

    # deconv
    CWF = blr.deconv_pmt(pmtrwf[event], coeff_c, coeff_blr,
                         pmt_active  = pmt_active,
                         n_baseline  = deconv_params.n_baseline,
                         thr_trigger = deconv_params.thr_trigger)

    # calibrated sum
    csum, csum_mau = cpf.calibrated_pmt_sum(CWF,
                                            adc_to_pes,
                                            pmt_active  = pmt_active,
                                            n_MAU       = 100,
                                            thr_MAU     = thr.thr_MAU)

    # zs sum
    s2_ene, s2_indx = cpf.wfzs(csum, threshold=thr.thr_s2)
    s1_ene, s1_indx = cpf.wfzs(csum_mau, threshold=thr.thr_s1)

    # S1 and S2
    S1 = cpf.find_S12(csum, s1_indx, **s1_params._asdict())
    S2 = cpf.find_S12(csum, s2_indx, **s2_params._asdict())

    #S2Si
    sipm = cpf.signal_sipm(sipmrwf[event], adc_to_pes_sipm,
                           thr=thr.thr_sipm, n_MAU=100)
    SIPM = cpf.select_sipm(sipm)
    S2Si = sipm_s2_dict(SIPM, S2, thr=thr.thr_SIPM)
    return (CalibratedSum(csum=csum, csum_mau=csum_mau),
            PMaps(S1=S1, S2=S2, S2Si=S2Si))
