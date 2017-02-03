"""Waveform to Data Frame Functions
This module includes functions that transform and manipulate raw waveforms
in data frames
authors: J.J. Gomez-Cadenas, G. Martinez
"""

import invisible_cities.core.wfm_functions as wfm
import pandas as pd

def get_waveforms_as_df(pmtea, event_number=0):
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


def get_energy_of_waveform_for_event_list_as_df(pmtea, event_list=[0]):
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
    swf['time_ns'] = time
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
    return df["time_ns"], df["ene_pes"]


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
    return wf2df(*wfm.rebin_wf(*df2wf(df), stride=stride))


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

def find_S12(wfzs, tmin=0*units.mus, tmax=1200*units.mus,
             stride=4, lmin=8, lmax=1e+6):
    """Find S1/S2 peaks.

    input: a zero supressed wf
    returns a list of waveform data frames
    do not interrupt the peak if next sample comes within stride
    accept the peak only if within [lmin, lmax)
    accept the peak only if within [tmin, tmax)
    """

    T = wfzs['time_ns'].values
    P = wfzs['ene_pes'].values

    S12 = {}
    pulse_on = 1
    j = 0

    S12[0] = [(T[0], P[0])]

    for i in range(1, len(wfzs)):
        if T[i] > tmax: break
        if T[i] < tmin: continue

        if wfzs.index[i] - stride > wfzs.index[i-1]:  #new s12
            j += 1
            S12[j] = [(T[i], P[i])]
        else:
            S12[j].append((T[i], P[i])

    S12L = []
    for s in S12:
        if lmin <= and len(s) < lmax:
            S12L.append(pd.DataFrame(s, columns=['time_ns','ene_pes']))
    return S12L
