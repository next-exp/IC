"""Waveform Functions
JJGC, September-October 2016

ChangeLog
12/10: change from time_ns to time_mus
"""

import math
import pandas as pd
import numpy as np

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


def get_waveforms(pmtea, event_number=0):
    """Produce a DataFrame with the waveforms in an array.

    Parameters
    ----------
    pmtea : tb.EArray
        The waveform (axis 2) for each sensor (axis 1) and event (axis 0).
    event_number : int
        Event number.

    Returns
    -------
    wfdf : pd.DataFrame
        Contains the waveform for each sensor indexed by ID.
    """
    NPMT = pmtea.shape[1]
    dic = {j: pmtea[event_number, j] for j in range(NPMT)}
    return pd.DataFrame(dic)


def get_waveforms_and_energy(pmtea, event_number=0):
    """Produce a DataFrame with the waveforms in an array and their energy.

    Parameters
    ----------
    pmtea : tb.EArray
        The waveform (axis 2) for each sensor (axis 1) and event (axis 0).
    event_number : int
        Event number.

    Returns
    -------
    wfdf : pd.DataFrame
        Contains the waveform for each sensor indexed by ID.
    wfe : pd.Series
        Contains the sum of the waveform for each sensor.
    """
    PMTWF = {}
    EPMT = []
    NPMT = pmtea.shape[1]

    for j in range(NPMT):
        # waveform for event event_number, PMT j
        PMTWF[j] = pmtea[event_number, j]
        epmt = np.sum(PMTWF[j])
        EPMT.append(epmt)
    return pd.DataFrame(PMTWF), pd.Series(EPMT)


def get_energy(pmtea, event_list=[0]):
    """Compute the sum of the waveforms for some events.

    Parameters
    ----------
    pmtea : tb.EArray
        The waveform (axis 2) for each sensor (axis 1) and event (axis 0).
    event_list : sequence of ints
        Event numbers.

    Returns
    -------
    wfes : pd.DataFrame
        Contains the sum of the waveform for each sensor indexed by ID.
    """
    NPMT = pmtea.shape[1]
    EPMT = []

    for i in event_list:
        epmt = np.zeros(NPMT)
        for j in range(NPMT):
            epmt[j] = np.sum(pmtea[i, j])
        EPMT.append(epmt)

    return pd.DataFrame(EPMT)

def wfdf(time,energy_pes):
    """Take two vectors (time, energy) and returns a data frame
    representing a waveform.
    """
    swf = {}
    swf['time_mus'] = time / units.mus
    swf['ene_pes'] = energy_pes
    return pd.DataFrame(swf)



def df2wf(df):
    """Retrieves the np arrays contained in the data frame.

    Parameters
    ----------
    df : pd.DataFrame
        Waveform data frame.

    Returns
    -------
    time_mus : 1-dim np.ndarray
        Waveform times.
    ene_pes : 1-dim np.ndarray
        Waveform amplitudes.
    """
    return df["time_mus"], df["ene_pes"]


def add_cwf(cwfdf, pmtdf):
    """Sum all PMTs for each time sample in pes.

    Parameters
    ----------
    cwfdf : pd.DataFrame
        A NPMT-column data frame holding the waveforms.
    pmtdf : pd.DataFrame
        Contains the sensors information.

    Returns
    -------
    swf : pd.DataFrame
        A data frame with the summed waveform.
    """
    summed = np.sum(to_pes(cwfdf.values.T, pmtdf), axis=1)
    idxs = np.arange(summed.size)
    return wf2df(idxs * 1, summed, idxs)


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


def rebin_df(df, stride=40):
    """Applies rebin_wf to a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame holding the waveform data.
    stride : int
        Integration step.

    Returns
    -------
    rebinned_df : pd.DataFrame
        Rebinned data frame.
    """
    return wf2df(*rebin_wf(*df2wf(df), stride=stride))


def wf_thr(df, threshold=1):
    """Get the values of a waveform above threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame holding the waveform.
    threshold : int or float
        Value from which values are ignored.

    Returns
    -------
    cutdf : pd.DataFrame
        Data frame holding the values surviving the cut.
    """
    return df[df.ene_pes > threshold]


def zs_wf(waveform, threshold, to_mus=None):
    """Remove waveform values below threshold.

    Parameters
    ----------
    waveform : 1-dim np.ndarray
        Waveform amplitudes.
    threshold : int or float
        Cut value.
    to_mus : int or float, optional
        Scale factor for converting times to microseconds. Default is None,
        meaning no conversion.

    Returns
    -------
    df : pd.DataFrame
        A two-field (time_mus and ene_pes) data frame holding the data
        surviving the cut.
    """
    t = np.argwhere(waveform > threshold).flatten()
    if not t.size:
        return None
    return wf2df(t if to_mus is None else t * to_mus, waveform[t])


def zero_suppression(waveforms, thresholds, to_mus=None):
    """Remove waveforms values below threshold.

    Parameters
    ----------
    waveforms : 2-dim np.ndarray
        Waveform amplitudes (axis 1) for each sensor (axis 0).
    thresholds : int, float or sequence of ints or floats
        Cut value for each sensors (sequence) or for all (single number).
    to_mus : int or float, optional
        Scale factor for converting times to microseconds. Default is None,
        meaning no conversion.

    Returns
    -------
    dfs : dictionary
        A dictionary holding two-field (time_mus and ene_pes) data frames
        containing the data surviving the cut. Keys are sensor IDs.
    """
    # If threshold is a single value, transform it into an array
    if not hasattr(thresholds, "__iter__"):
        thresholds = np.ones(waveforms.shape[0]) * thresholds
    zsdata = map(zs_wf, waveforms, thresholds)
    return {i: df for i, df in enumerate(zsdata) if df is not None}


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
