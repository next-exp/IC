"""PMAPS functions.
JJGC December 2016

"""
import numpy  as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt

from .. icaro.mpl_functions     import circles
from .. icaro.mpl_functions     import set_plot_labels
from .. core.system_of_units_c import units

from .. database               import load_db

from .  pmaps_functions_c      import df_to_pmaps_dict
from .  pmaps_functions_c      import df_to_s2si_dict

def load_pmaps(PMP_file_name):
    """Read the PMAP file and return transient PMAP rep."""

    s1t, s2t, s2sit = read_pmaps(PMP_file_name)
    S1              = df_to_pmaps_dict(s1t)
    S2              = df_to_pmaps_dict(s2t)
    S2Si            = df_to_s2si_dict(s2sit)
    return S1, S2, S2Si


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
    """Print  the peaks of input S12.

    S12 is a dictionary
    S12[i] for i in keys() are the S12 peaks
    """
    print('number of peaks = {}'.format(len(S12)))
    for i in S12.keys():
        print('S12 number = {}, samples = {} sum in pes ={}'
              .format(i, len(S12[i][0]), np.sum(S12[i][1])))


def plot_s12(S12, figsize=(6,6)):
    """Plot the peaks of input S12.

    S12 is a dictionary
    S12[i] for i in keys() are the S12 peaks
    """
    plt.figure(figsize=figsize)

    set_plot_labels(xlabel = "t (mus)",
                    ylabel = "S12 (pes)")
    xy = len(S12)
    if xy == 1:
        t = S12[0][0]
        E = S12[0][1]
        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(t/units.mus, E)
    else:
        x = 3
        y = xy/x
        if y % xy != 0:
            y = int(xy/x) + 1
        for i in S12.keys():
            ax1 = plt.subplot(x, y, i+1)
            t = S12[i][0]
            E = S12[i][1]
            plt.plot(t/units.mus, E)


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


def width(times, to_mus=False):
    """
    Compute peak width. Times has to be ordered.
    """

    w = times[-1] - times[0] if len(times) > 0 else 0
    return w * units.ns/units.mus if to_mus else w


def integrate_sipm_charges_in_peak_as_dict(Si):
    """Return dict of integrated charges from a SiPM dictionary.

    S2Si = {  peak : Si }
      Si = { nsipm : [ q1, q2, ...] }

    Returns an integrated Si = { nsipm : sum(q_n) }
    """
    return { sipm : sum(qs) for (sipm, qs) in Si.items() }


def integrate_sipm_charges_in_peak(Si):
    """Return arrays of nsipm and integrated charges from SiPM dictionary.

    S2Si = {  peak : Si }
      Si = { nsipm : [ q1, q2, ...] }

    Returns (np.array[[nsipm_1 ,     nsipm_2, ...]],
             np.array[[sum(q_1), sum(nsipm_2), ...]])
    """
    sipms_and_Q_totals = integrate_sipm_charges_in_peak_as_dict(Si)
    sipms = np.array(tuple(sipms_and_Q_totals.keys()))
    Qs    = np.array(tuple(sipms_and_Q_totals.values()))
    return sipms, Qs


def integrate_S2Si_charge(S2Si):
    """Return S2Si containing integrated charges.

    S2Si = {  peak : Si }
      Si = { nsipm : [ q1, q2, ...] }

    Returns S2Si where Si = { nsipm : sum([q1, q2, ...])}
"""
    return { peak_no : integrate_sipm_charges_in_peak_as_dict(peak)
             for (peak_no, peak) in S2Si.items() }


def select_si_slice(si, slice_no):
    # This is a temporary fix! The number of slices in the SiPM arrays
    # must match that of the PMT PMaps.
    return {sipm_no: (sipm[slice_no] if len(sipm) > slice_no else 0)
                      for sipm_no, sipm in si.items()}
