"""PMAPS plotting functions.

"""
import numpy  as np
import pandas as pd
import tables as tb
import matplotlib.pyplot as plt

from .  mpl_functions     import circles
from .  mpl_functions     import set_plot_labels
from .. core.system_of_units_c import units

from .. database               import load_db



def plot_s12(s12, figsize=(6,6)):
    """Plot the peaks of input S12.

    Uses the s12 interface defined in event model
    """
    plt.figure(figsize=figsize)

    set_plot_labels(xlabel = "t (mus)",
                    ylabel = "S12 (pes)")
    xy = s12.number_of_peaks
    if xy == 1:
        wfm = s12.peak_waveform(0)
        ax1 = plt.subplot(1, 1, 1)
        ax1.plot(wfm.t/units.mus, wfm.E)
    else:
        x = 3
        y = xy/x
        if y % xy != 0:
            y = int(xy/x) + 1
        for i in range(xy):
            ax1 = plt.subplot(x, y, i+1)
            wfm = s12.peak_waveform(i)
            plt.plot(wfm.t/units.mus, wfm.E)


def plot_s2si_map(s2si, cmap='Blues'):
        """Plot a map of the energies of S2Si objects."""

        DataSensor = load_db.DataSiPM(0)
        radius = 2
        xs = DataSensor.X.values
        ys = DataSensor.Y.values
        r = np.ones(len(xs)) * radius
        #col = np.zeros(len(xs))

        col = np.array([s2si.sipm_total_energy(peak_no, sipm_no)
               for peak_no in range(s2si.number_of_peaks)
               for sipm_no in s2si.sipms_in_peak(peak_no)])
        plt.figure(figsize=(8, 8))
        plt.subplot(aspect="equal")
        circles(xs, ys, r, c=col, alpha=0.5, ec="none", cmap=cmap)
        plt.colorbar()

        plt.xlim(-198, 198)
        plt.ylim(-198, 198)
