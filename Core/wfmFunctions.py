"""Waveform Functions
JJGC, September-October 2016

ChangeLog
12/10: change from time_ns to time_mus
"""

import math
import pandas as pd
import numpy as np

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
    suppressed_wfs = map(suppress_wf, waveforms, thresholds)
    return np.array(suppressed_wfs)


def find_baseline(waveform, n_samples=500, check_no_signal=True):
    """Find baseline in waveform.

    Parameters
    ----------
    waveform : 1-dim np.ndarray
        Any sensor's waveform.

    n_samples : int, optional
        Number of samples to measure baseline. Default is 500.

    check_no_signal : bool, optional
        Check RMS in waveform subsample to ensure there is no signal present
        in it. Default is True.

    Returns
    -------
    baseline : int or float
        Waveform's baseline.
    """
    if check_no_signal:
        for i in range(waveform.size//n_samples):
            low = i * n_samples
            upp = low + n_samples
            subsample = waveform[low:upp]
            if np.std(subsample) < 3:
                return np.mean(subsample)
    return np.mean(waveform[:n_samples])


def subtract_baseline(waveforms, n_samples=500, check_no_signal=True):
    """Compute the baseline for each sensor in the event and subtract it.

    Parameters
    ----------
    waveforms : 2-dim np.ndarray
        The waveform amplitudes (axis 1) for each sensor (axis 0)
    n_samples : int
        Number of samples to measure baseline. Default is 500.
    check_no_signal : bool, optional
        Check RMS in waveform subsample to ensure there is no signal present
        in it. Default is True.

    Returns
    -------
    blr_wfs : 2-dim np.array
        The input waveform with the baseline subtracted.
    """
    bls = np.apply_along_axis(lambda wf: find_baseline(wf, n_samples,
                                                       check_no_signal),
                              1, waveforms)
    return waveforms - bls.reshape(waveforms.shape[0], 1)
