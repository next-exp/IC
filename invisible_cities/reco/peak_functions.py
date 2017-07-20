"""Functions to find peaks, S12 selection etc.
Last revision: JJGC, July, 2017

A python module containing peak finding functions. Most of the functions
are private and used only for testing (e.g, comparison with the
corresponding cython functions). Private functions are labelled _function_name
the corresponding public function (to be found in the cython module)
is labelled function_name.

"""


import numpy  as np

from scipy import signal

from .. core   import system_of_units as units
from .. sierpe import blr

from ..io                  import pmap_io as pio
from .                     import peak_functions_c as cpf
from .. evm.ic_containers  import CSum
from .. evm.ic_containers  import PMaps
from .. types.ic_types     import minmax

def _calibrated_pmt_sum(CWF, adc_to_pes, pmt_active = [], n_MAU=200, thr_MAU=5):
    """Compute the ZS calibrated sum of the PMTs
    after correcting the baseline with a MAU to suppress low frequency noise.
    input:
    CWF         : Corrected waveform (passed by BLR)
    adc_to_pes  : a vector with calibration constants
    n_MAU       : length of the MAU window
    thr_MAU     : treshold above MAU to select sample

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

    Parameters:
    t_s:       starting time for waveform
    t_finish:  end time for waveform
    wf:        wavform (chunk)
    stride:    How many (25 ns) samples we combine into a single bin
    """

    assert (ts < t_finish)
    bs = 25*units.ns  # bin size
    rbs = bs * stride # rebinned bin size

    # Find the nearest time (in stride samples) before ts
    t_start  = int((ts // (rbs)) * rbs)
    t_total  = t_finish - t_start
    n = int(t_total // (rbs))  # number of samples
    r = int(t_total  % (rbs))

    lenb = n
    if r > 0: lenb = n+1

    T = np.zeros(lenb, dtype=np.double)
    E = np.zeros(lenb, dtype=np.double)

    j = 0
    for i in range(n):
        esum  = 0
        for tb in range(int(t_start +  i   *rbs),
                        int(t_start + (i+1)*rbs),
                        int(bs)):
            if tb < ts: continue
            esum += wf[j]
            j    += 1

        E[i] = esum
        if i == 0: T[0] = np.mean((ts, t_start + rbs))
        else     : T[i] = t_start + i*rbs + rbs/2

    if r > 0:
        esum  = 0
        for tb in range(int(t_start + n*rbs),
                        int(t_finish),
                        int(bs)):
            if tb < ts: continue
            esum += wf[j]
            j    += 1

        E[n] = esum
        if n == 0: T[n] = np.mean((ts             , t_finish))
        else     : T[n] = np.mean((t_start + n*rbs, t_finish))

    assert j == len(wf) # ensures you have rebinned correctly the waveform
    return T, E


def _find_s12(csum, index,
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

    """

    T = cpf._time_from_index(index)

    S12  = {}
    S12L = {}

    # Start end end index of S12, [start i, end i)
    S12[0] = np.array([index[0], index[0] + 1], dtype=np.int32)

    # Start end end index of S12, [start i, end i)
    i_min = int(time[0] / (25*units.ns))     # index in csum  corresponding to t.min
    i_i = np.where(index >= i_min)[0].min()  # index in index corresponding to t.min (or first
                                             # time not threshold suppressed)
    S12[0] = np.array([index[i_i], index[i_i]+ 1], dtype=np.int32)

    j = 0
    for i in range(i_i + 1, len(index)):
        assert T[i] > time[0]
        if T[i] > time.max: break
        # New s12, create new start and end index
        elif index[i] - stride > index[i-1]:
            j += 1
            S12[j] = np.array([index[i], index[i] + 1], dtype=np.int32)
        # Update end index in current S12
        else: S12[j][1] = index[i] + 1

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

    return S12L
    #return pio.S12(S12L)


def _sipm_s2_dict(SIPM, S2d, thr=5 * units.pes):
    """Given a vector with SIPMs (energies above threshold), and a
    dictionary of S2s, S2d, returns a dictionary of SiPMs-S2.  Each
    index of the dictionary correspond to one S2 and is a list of np
    arrays. Each element of the list is the S2 window in the SiPM (if
    not zero)
    """
    return {i: _sipm_s2(SIPM, s2l, thr=thr) for i, s2l in S2d.items()}


def _sipm_s2(dSIPM, s2l, thr=5*units.pes):
    """Given a vector with SIPMs (energies above threshold), and a list containing s2 times
    and energies, return a dict of np arrays, where the key is the sipm with signal.
    """

    def index_from_s2(s2l):
        """Return the indexes defining the vector."""
        t0 = int(s2l[0][0] // units.mus)
        return t0, t0 + len(s2l[0])

    i0, i1 = index_from_s2(s2l)
    SIPML = {}
    for ID, sipm in dSIPM.values():
        slices = sipm[i0:i1]
        psum = np.sum(slices)
        if psum > thr:
            SIPML[ID] = slices.astype(np.double)
    return SIPML


def compute_csum_and_pmaps(event, pmtrwf, sipmrwf,
                           s1par, s2par, thresholds,
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

    s1 = cpf.find_s1(csum,
                      s1_indx,
                      **s1_params._asdict())

    s2 = cpf.find_s2(csum,
                      s2_indx,
                      **s2_params._asdict())

    s1 = cpf.correct_s1_ene(s1.s1d, csum)
    sipmzs = cpf.signal_sipm(sipmrwf[event], adc_to_pes_sipm,
                           thr=thr.thr_sipm, n_MAU=100)

    s2si = cpf.find_s2si(sipmzs, s2.s2d, thr = thr.thr_SIPM)


    return (CSum(csum=csum, csum_mau=csum_mau),
            PMaps(S1=s1.s1d, S2=s2.s2d, S2Si=s2si.s2sid))
