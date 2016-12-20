"""PMAPS functions.
JJGC December 2016

"""
from __future__ import print_function

import math
import numpy as np
import pandas as pd
import tables as tb
import Core.system_of_units as units
import ICython.Core.pmapsFunctions as cpm
import matplotlib.pyplot as plt


def read_PMAPS(path, pmap_file):
    """Return the PMAPS as PD DataFrames."""

    h5f = tb.open_file(path+pmap_file,'r+')
    s1t = h5f.root.PMAPS.S1
    s2t = h5f.root.PMAPS.S2
    s2sit = h5f.root.PMAPS.S2Si

    return pd.DataFrame.from_records(s1t.read()),\
           pd.DataFrame.from_records(s2t.read()),\
           pd.DataFrame.from_records(s2sit.read())


def S12_select_event(S12df, event):
    """Return a copy of the DF for event."""
    return S12df.loc[lambda df: df.event.values ==event, :]


def S12_select_peak(S12df, peak):
    """Return a copy of the DF for peak."""
    return S12df.loc[lambda df: S12df.peak.values ==peak, :]


def S12_get_wvfm(S12df, event, peak):
    """Return the waveform for event, peak."""
    s12t = S12_select_event(S12df, event)
    s12p = S12_select_peak(s12t, peak=peak)
    return s12p.time.values, s12p.ene.values


def S12df_to_dict(s12df, evt_max=10):
    """
    iput: S12df object (a S12 pytable readout as a PD dataframe)
    returns: an S12L dictionary
    """
    peak = s12df.peak.values.astype(np.int32)

    return cpm.cdf_to_dict(len(s12df.index), evt_max, s12df.event.values,
                           peak, s12df.time.values, s12df.ene.values)



def plot_S12df_waveforms(S12df, nmin=0, nmax=16, x=4, y=4):
    """Take as input a S1df PMAPS and plot waveforms."""
    plt.figure(figsize=(12, 12))

    for i in range(nmin, nmax):
        plt.subplot(y, y, i+1)
        T, E = S12_get_wvfm(S12df, event=i, peak=0)
        plt.plot(T, E)
    plt.show()
