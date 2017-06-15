"""Functions to find peaks, S12 selection etc.
JJGC and GML December 2016
"""


import numpy  as np

from scipy import signal

from .. core   import system_of_units as units
from .. sierpe import blr

from .         import peak_functions_c as cpf
from .  params import CalibratedSum
from .  params import PMaps
from .. core.ic_types          import minmax

def calibrated_pmt_sum(CWF, adc_to_pes, pmt_active = [], n_MAU=200, thr_MAU=5):
    """Compute the ZS calibrated sum of the PMTs
    after correcting the baseline with a MAU to suppress low frequency noise.
    input:
    CWF         : Corrected waveform (passed by BLR)
    adc_to_pes  : a vector with calibration constants
    n_MAU       : length of the MAU window
    thr_MAU     : treshold above MAU to select sample

    NB: This function is used mainly for testing purposes. It is
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


def wfzs(wf, threshold=0):
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

    NB: This function is used mainly for testing purposed. It is
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


def time_from_index(indx):
    """Return the times (in ns) corresponding to the indexes in indx

    NB: This function is used mainly for testing purposed. It is
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


def rebin_waveform(t, e, stride=40):
    """
    Rebin a waveform according to stride
    The input waveform is a vector such that the index expresses time bin and the
    contents expresses energy (e.g, in pes)
    The function returns the rebinned T and E vectors

    NB: This function is used mainly for testing purposed. It is
     programmed "c-style", which is not necesarily optimal
    in python, but follows the same logic that the corresponding cython
    function (in peak_functions_c), which runs faster and should be used
    instead of this one for nornal calculations.

    """

    assert len(t) == len(e)

    n = len(t) // stride
    r = len(t) %  stride

    lenb = n
    if r > 0:
        lenb = n+1

    T = np.zeros(lenb, dtype=np.double)
    E = np.zeros(lenb, dtype=np.double)

    j = 0
    for i in range(n):
        esum = 0
        tmean = 0
        for k in range(j, j + stride):
            esum  += e[k]
            tmean += t[k]

        tmean /= stride
        E[i] = esum
        T[i] = tmean
        j += stride

    if r > 0:
        esum  = 0
        tmean = 0
        for k in range(j, len(t)):
            esum  += e[k]
            tmean += t[k]
        tmean /= (len(t) - j)
        E[n] = esum
        T[n] = tmean

    return T, E


def find_S12(wfzs, index,
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

    NB: This function is a wrapper around the cython function. It returns
    a dictionary of namedtuples (Waveform(t = [t], E = [E])), where
    [t] and [E] are np arrays.
    """

    from collections import namedtuple

    Waveform = namedtuple('Waveform', 't E')

    S12 = cpf.find_S12(wfzs, index,
                       *t, *l,
                      stride,
                      rebin, rebin_stride)

    return {i: Waveform(t, E) for i, (t,E) in S12.items()}

def find_S12_py(wfzs, index,
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

    P = wfzs
    T = time_from_index(index)

    assert len(wfzs) == len(index)

    S12  = {}
    S12L = {}
    s12  = []

    S12[0] = s12
    S12[0].append([T[0], P[0]])

    j = 0
    for i in range(1, len(wfzs)) :

        if T[i] > time.max:
            break

        if T[i] < time.min:
            continue

        if index[i] - stride > index[i-1]:  #new s12
            j += 1
            s12 = []
            S12[j] = s12
        S12[j].append([T[i], P[i]])

    # re-arrange and rebin
    j = 0
    for i in S12:
        ls = len(S12[i])

        if not (length.min <= ls < length.max):
            continue

        t = np.zeros(ls, dtype=np.double)
        e = np.zeros(ls, dtype=np.double)

        for k in range(ls):
            t[k] = S12[i][k][0]
            e[k] = S12[i][k][1]

        if rebin == True:
            TR, ER = rebin_waveform(t, e, stride = rebin_stride)
            S12L[j] = [TR, ER]
        else:
            S12L[j] = [t, e]
        j += 1

    return S12L


def sipm_s2_dict(SIPM, S2d, thr=5 * units.pes):
    """Given a vector with SIPMs (energies above threshold), and a
    dictionary of S2s, S2d, returns a dictionary of SiPMs-S2.  Each
    index of the dictionary correspond to one S2 and is a list of np
    arrays. Each element of the list is the S2 window in the SiPM (if
    not zero)
    """
    return {i: sipm_s2(SIPM, S2, thr=thr) for i, S2 in S2d.items()}


def index_from_s2(S2):
    """Return the indexes defining the vector."""
    t0 = int(S2[0][0] // units.mus)
    return t0, t0 + len(S2[0])


def sipm_s2(dSIPM, S2, thr=5*units.pes):
    """Given a vector with SIPMs (energies above threshold), return a dict
    of np arrays, where the key is the sipm with signal.
    """
    #import pdb; pdb.set_trace()

    i0, i1 = index_from_s2(S2)
    SIPML = {}
    for ID, sipm in dSIPM.values():
        slices = sipm[i0:i1]
        psum = np.sum(slices)
        if psum > thr:
            SIPML[ID] = slices.astype(np.double)
    return SIPML


def compute_csum_and_pmaps(pmtrwf, sipmrwf, s1par, s2par, thresholds,
                           event, calib_vectors, deconv_params):
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
    S1 = cpf.find_S12(s1_ene, s1_indx, **s1_params._asdict())
    S2 = cpf.find_S12(s2_ene, s2_indx, **s2_params._asdict())

    #S2Si
    sipm = cpf.signal_sipm(sipmrwf[event], adc_to_pes_sipm,
                           thr=thr.thr_sipm, n_MAU=100)
    SIPM = cpf.select_sipm(sipm)
    S2Si = sipm_s2_dict(SIPM, S2, thr=thr.thr_SIPM)
    return (CalibratedSum(csum=csum, csum_mau=csum_mau),
            PMaps(S1=S1, S2=S2, S2Si=S2Si))


def select_peaks(peaks,
                 Emin, Emax,
                 Lmin, Lmax,
                 Hmin, Hmax,
                 Ethr = -1):

    is_valid = lambda E: (Lmin <= np.size(E) < Lmax and
                          Hmin <= np.max (E) < Hmax and
                          Emin <= np.sum (E) < Emax)

    return {peak_no: (t, E) for peak_no, (t, E) in peaks.items() if is_valid(E[E > Ethr])}


def select_Si(peaks,
              Nmin, Nmax):
    is_valid = lambda sipms: Nmin <= len(sipms) < Nmax
    return {peak_no: sipms for peak_no, sipms in peaks.items() if is_valid(sipms)}
