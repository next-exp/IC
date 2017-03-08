"""PMAPS/DF functions.
PMAPS functions involving DFs.

Not used in main stream analysis. Not tested.
Kept in repository for future use.

"""
from __future__ import print_function, division, absolute_import

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def s12df_select_event(S12df, event):
    """Return a copy of the s12df for event."""
    return S12df.loc[lambda df: df.event.values == event, :]


def s12df_select_peak(S12df, peak):
    """Return a copy of the s12df for peak."""
    return S12df.loc[lambda df: S12df.peak.values == peak, :]


def s12df_get_wvfm(S12df, event, peak):
    """Return the waveform for event, peak."""
    s12t = s12df_select_event(S12df, event)
    s12p = s12df_select_peak(s12t, peak)
    return s12p.time.values, s12p.ene.values


def s12df_plot_waveforms(S12df, nmin=0, nmax=16, x=4, y=4):
    """Take as input a S1df PMAPS and plot waveforms."""
    plt.figure(figsize=(12, 12))

    for i in range(nmin, nmax):
        plt.subplot(y, y, i+1)
        T, E = s12df_get_wvfm(S12df, event=i, peak=0)
        plt.plot(T, E)


def s12_to_wvfm_list(S12):
    """Take an S12 dictionary and return a list of DF."""
    S12L = []
    print('number of peaks = {}'.format(len(S12)))
    for i in S12.keys():
        s12df = pd.DataFrame(S12[i], columns=['time_ns','ene_pes'])
        print('S12 number = {}, samples = {} sum in pes ={}'
              .format(i, len(s12df), np.sum(s12df.ene_pes.values)))
        S12L.append(s12df)
    return S12L


def scan_s12l(S12L):
    """Print and plot the peaks of input list S12L S12L is a list of data
    frames.
    """
    print('number of peaks = {}'.format(len(S12L)))
    for i, s12df in enumerate(S12L):
        print('S12 number = {}, samples = {} sum in pes ={}'
              .format(i, len(s12df), np.sum(s12df.ene_pes.values)))
        plt.plot(s12df.time_ns.values, s12df.ene_pes)
        plt.show()
        raw_input('hit return')
