"""PMAPS functions.
JJGC December 2016

"""
from __future__ import print_function, division, absolute_import

import math
import numpy as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt
import invisible_cities.core.system_of_units as units
import invisible_cities.reco.pmaps_functions_c as cpm
import invisible_cities.core.core_functions as cf
from   invisible_cities.database import load_db
from   invisible_cities.core.mpl_functions import circles



def read_pmaps(PMP_file):
    """Return the PMAPS as PD DataFrames."""
    with tb.open_file(PMP_file, 'r') as h5f:
        s1t = h5f.root.PMAPS.S1
        s2t = h5f.root.PMAPS.S2
        s2sit = h5f.root.PMAPS.S2Si

        return (pd.DataFrame.from_records(s1t  .read()),
                pd.DataFrame.from_records(s2t  .read()),
                pd.DataFrame.from_records(s2sit.read()))


def s12df_to_s12l(s12df, evt_max=None):
    """
    Accept a S12df object (a S12 pytable readout as a PD dataframe)
    and return a S12L dictionary
    """
    peak = s12df.peak.values.astype(np.int32)
    if evt_max is None:
        evt_max = -1
    return cpm.cdf_to_dict(len(s12df.index), evt_max, s12df.event.values,
                           peak, s12df.time.values,   s12df.ene  .values)


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


def scan_s12(S12):
    """Print and plot the peaks of input S12.

    S12 is a dictionary
    S12[i] for i in keys() are the S12 peaks
    """
    print('number of peaks = {}'.format(len(S12)))
    for i in S12.keys():
        print('S12 number = {}, samples = {} sum in pes ={}'
              .format(i, len(S12[i][0]), np.sum(S12[i][1])))
        plt.plot(            S12[i][0],         S12[i][1])


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


def plot_s2si_map(S2Si, cmap='Blues'):
        """Plot a map of the energies of S2Si objects."""

        DataSensor = load_db.DataSiPM(0)
        radius = 2
        xs = DataSensor.X.values
        ys = DataSensor.Y.values
        r = np.ones(len(xs)) * radius
        col = np.zeros(len(xs))
        for k in S2Si:
            s2si = S2Si[k]
            for e in s2si:
                indx = e[0]
                ene = np.sum(e[1])
                col[indx] = ene
        plt.figure(figsize=(8, 8))
        plt.subplot(aspect="equal")
        circles(xs, ys, r, c=col, alpha=0.5, ec="none", cmap=cmap)
        plt.colorbar()

        plt.xlim(-198, 198)
        plt.ylim(-198, 198)


def scan_s2si_map(S2Si):
        """Scan the S2Si objects."""

        for k in S2Si:
            s2si = S2Si[k]
            for e in s2si:
                indx = e[0]
                ene = np.sum(e[1])
                print('SiPM number = {}, total energy = {}'.format(indx, ene))


class S12F:
    """
    Defines the global features of an S12 peak, namely:
    1) peak width
    2) peak maximum (both energy and time)
    3) energy total
    4) ratio peak/total energy
    """

    def __init__(self, length):
        self.w    = np.zeros(length, dtype=np.double)
        self.tmax = np.zeros(length, dtype=np.double)
        self.emax = np.zeros(length, dtype=np.double)
        self.etot = np.zeros(length, dtype=np.double)
        self.er   = np.zeros(length, dtype=np.double)


def s12_features(S12L, peak=0, max_events=100):
    """
    input: S1L
    returns a S1F object for specific peak
    """
    nk = np.array(list(S12L.keys()))

    # max event: but notice that some events may be missing
    evt_max = np.max(nk)

    n = min(evt_max+1, max_events)
    print('required {} events; found in dict {} events'
          .format(max_events, evt_max+1))
    s1f = S12F(n)

    for i in nk:
        if i >= n:
            break

        S1 = S12L[i]
        try:
            T = S1[peak][0]
            E = S1[peak][1]
        except KeyError:
            print('peak number {} does not exit in S12L'.format(peak))
            return 0

        s1f.w[i] = T[-1] - T[0]
        s1f.emax[i] = np.max(E)
        i_t = cf.loc_elem_1d(E, s1f.emax[i])
        s1f.tmax[i] = T[i_t]
        s1f.etot[i] = np.sum(E)

        if s1f.etot[i] > 0:
            s1f.er[i] = s1f.emax[i] / s1f.etot[i]
        else:
            s1f.er[i] = 0

    return s1f
