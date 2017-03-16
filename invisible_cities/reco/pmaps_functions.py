"""PMAPS functions.
JJGC December 2016

"""
from __future__ import print_function, division, absolute_import

import numpy as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt
from   invisible_cities.reco.pmaps_functions_c import df_to_pmaps_dict, df_to_s2si_dict
import invisible_cities.core.core_functions as cf
from   invisible_cities.database import load_db
from   invisible_cities.core.mpl_functions import circles


def read_pmaps(PMP_file_name):
    """Return the PMAPS as PD DataFrames."""
    with tb.open_file(PMP_file_name, 'r') as h5f:
        s1t   = h5f.root.PMAPS.S1
        s2t   = h5f.root.PMAPS.S2
        s2sit = h5f.root.PMAPS.S2Si

        return (pd.DataFrame.from_records(s1t  .read()),
                pd.DataFrame.from_records(s2t  .read()),
                pd.DataFrame.from_records(s2sit.read()))

def read_run_and_event_from_pmaps_file(PMP_file_name):
    """Return the PMAPS as PD DataFrames."""
    with tb.open_file(PMP_file_name, 'r') as h5f:
        event_t = h5f.root.Run.events
        run_t   = h5f.root.Run.runInfo

        return (pd.DataFrame.from_records(run_t  .read()),
                pd.DataFrame.from_records(event_t.read()))


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


def plot_s2si_map(S2Si, cmap='Blues'):
        """Plot a map of the energies of S2Si objects."""

        DataSensor = load_db.DataSiPM(0)
        radius = 2
        xs = DataSensor.X.values
        ys = DataSensor.Y.values
        r = np.ones(len(xs)) * radius
        col = np.zeros(len(xs))
        for sipm in S2Si.values():
            for nsipm, E in sipm.items():
                ene = np.sum(E)
                col[nsipm] = ene
        plt.figure(figsize=(8, 8))
        plt.subplot(aspect="equal")
        circles(xs, ys, r, c=col, alpha=0.5, ec="none", cmap=cmap)
        plt.colorbar()

        plt.xlim(-198, 198)
        plt.ylim(-198, 198)

def scan_s2si_map(S2Si):
    """Scan the S2Si objects."""
    for sipm in S2Si.values():
        for nsipm, E in sipm.items():
            ene = np.sum(E)
            print('SiPM number = {}, total energy = {}'.format(nsipm, ene))


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
