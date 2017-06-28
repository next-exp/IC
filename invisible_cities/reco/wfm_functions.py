"""Waveform Functions
This module includes functions to manipulate waveforms.
authors: J.J. Gomez-Cadenas, G. Martinez
"""

import math

import numpy as np

import matplotlib.pyplot as plt

from .. core.core_functions import define_window
from .. sierpe              import blr

def to_adc(wfs, adc_to_pes):
    """
    Convert waveform in pes to adc.

    Parameters
    ----------
    wfs : 2-dim np.ndarray
        The waveform (axis 1) for each sensor (axis 0).
    sensdf : a vector of constants
        Contains the sensor-related information.

    Returns
    -------
    adc_wfs : 2-dim np.ndarray
        The input wfs scaled to adc.
    """
    return wfs * adc_to_pes.reshape(wfs.shape[0], 1)


def to_pes(wfs, adc_to_pes):
    """
    Convert waveform in adc to pes.

    Parameters
    ----------
    wfs : 2-dim np.ndarray
        The waveform (axis 1) for each sensor (axis 0).
    sensdf : a vector of constants
        Contains the sensor-related information.

    Returns
    -------
    pes_wfs : 2-dim np.ndarray
        The input wfs scaled to pes.
    """
    return wfs / adc_to_pes.reshape(wfs.shape[0], 1)

## TODO DECIDE WHAT TO DO WITH THESE REBIN FUNCTIONS: (replace, delete or move)
def rebin_wf(t, e, stride=40):
    """Rebin arrays according to some stride.

    Parameters
    ----------
    t : np.ndarray
        Array of times.
    e : np.ndarray
        Array of amplitudes.
    stride : int
        Integration step.

    Returns
    -------
    rebinned_t : np.ndarray
        Rebinned array of times.
    rebinned_e : np.ndarray
        Rebinned array of amplitudes.

    NB: SLOW, see coreFunctions_perf.py in this directory
    """
    n = int(math.ceil(len(t) / float(stride)))
    T = np.empty(n, dtype=np.float32)
    E = np.empty(n, dtype=np.float32)

    for i in range(n):
        low = i * stride
        upp = low + stride
        E[i] = np.sum(e[low:upp])
        T[i] = np.mean(t[low:upp])

    return T, E

def rebin_waveform(t, e, stride = 40):
    """Rebin waveforms t and e according to stride.

    Parameters
    ----------
    waveforms t,e : 1-dim np.ndarrays representing time and energy vectors

    Returns
    -------
    rebinned waveforms : 1-dim np.ndarrays rebinned according to stride
        A copy of the input waveform with values below threshold set to zero.

    NB
    ---
    Rebin function uses loops, which makes it very slow, but straight forward
    extensible to the cython version.

    10 times faster than rebin_wf
    """

    assert(len(t) == len(e))

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
        esum = 0
        tmean = 0
        for k in range(j, len(t)):
            esum  += e[k]
            tmean += t[k]
        tmean /= (len(t) - j)
        E[n] = esum
        T[n] = tmean


    return T, E


def suppress_wf(waveform, threshold):
    """Put zeros where the waveform is below some threshold.

    Parameters
    ----------
    waveform : 1-dim np.ndarray
        Waveform amplitudes.
    threshold : int or float
        Cut value.

    Returns
    -------
    wf : 1-dim np.ndarray
        A copy of the input waveform with values below threshold set to zero.
    """
    wf = np.copy(waveform)
    wf[wf <= threshold] = 0
    return wf


def noise_suppression(waveforms, thresholds):
    """Put zeros where the waveform is below some threshold.

    Parameters
    ----------
    waveforms : 2-dim np.ndarray
        Waveform amplitudes (axis 1) for each sensor (axis 0).
    thresholds : int or float or sequence of ints or floats
        Cut value for each waveform (sequence) or for all (single number).

    Returns
    -------
    suppressed_wfs : 2-dim np.ndarray
        A copy of the input waveform with values below threshold set to zero.
    """
    if not hasattr(thresholds, "__iter__"):
        thresholds = np.ones(waveforms.shape[0]) * thresholds
    suppressed_wfs = list(map(suppress_wf, waveforms, thresholds))
    return np.array(suppressed_wfs)


def cwf_from_rwf(pmtrwf, event_list, calib_vectors, deconv_params):
    """Compute CWF from RWF for a given list of events."""

    CWF=[]
    for event in event_list:
        CWF.append(blr.deconv_pmt(pmtrwf[event], calib_vectors.coeff_c,
                             calib_vectors.coeff_blr,
                             n_baseline=deconv_params.n_baseline,
                             thr_trigger=deconv_params.thr_trigger))
    return CWF


def compare_cwf_blr(cwf, pmtblr, event_list, window_size=500):
    """Return differences between the area of the CWF and the BLR in a
    given window, for a list of events, expressed as a percentage.
    """
    DIFF = []
    for event in event_list:
        CWF = cwf[event]
        BLR = pmtblr[event]

        for i in range(len(BLR)):
            t0, t1 = define_window(BLR[i], window_size)
            diff = abs(np.sum(BLR[i][t0:t1]) - np.sum(CWF[i][t0:t1]))
            diff = 100. * diff / np.sum(BLR)
            DIFF.append(diff)

    return np.array(DIFF)

def plot_waveform(pmtwf, zoom=False, window_size=800):
    """Take as input a vector a single waveform and plot it"""

    first, last = 0, len(pmtwf)
    if zoom:
        first, last = define_window(pmtwf, window_size)

    mpl.set_plot_labels(xlabel="samples", ylabel="adc")
    plt.plot(pmtwf[first:last])


def plot_pmt_waveforms(pmtwfdf, zoom=False, window_size=800):
    """Take as input a vector storing the PMT wf and plot the waveforms"""
    plt.figure(figsize=(12, 12))
    for i in range(len(pmtwfdf)):
        first, last = 0, len(pmtwfdf[i])
        if zoom:
            first, last = define_window(pmtwfdf[i], window_size)
        plt.subplot(3, 4, i+1)
        # ax1.set_xlim([0, len_pmt])
        mpl.set_plot_labels(xlabel="samples", ylabel="adc")
        plt.plot(pmtwfdf[i][first:last])


def plot_waveforms_overlap(wfs, zoom=False, window_size=800):
    """Draw all waveforms together. If zoom is True, plot is zoomed
    around peak.
    """
    first, last = 0, wfs.shape[1]
    if zoom:
        first, last = define_window(wfs[0], window_size)
    for wf in wfs:
        plt.plot(wf[first:last])


def plot_wfa_wfb(wfa, wfb, zoom=False, window_size=800):
    """Plot together wfa and wfb, where wfa and wfb can be
    RWF, CWF, BLR.
    """
    plt.figure(figsize=(12, 12))
    for i in range(len(wfa)):
        first, last = 0, len(wfa[i])
        if zoom:
            first, last = define_window(wfa[i], window_size)
        plt.subplot(3, 4, i+1)
        # ax1.set_xlim([0, len_pmt])
        mpl.set_plot_labels(xlabel="samples", ylabel="adc")
        plt.plot(wfa[i][first:last], label= 'WFA')
        plt.plot(wfb[i][first:last], label= 'WFB')
        legend = plt.legend(loc='upper right')
        for label in legend.get_texts():
            label.set_fontsize('small')
