"""A utility module for plots with matplotlib"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show

from .. core.system_of_units_c import units
from .. core.core_functions import define_window
from .. database     import load_db
from . mpl_functions import set_plot_labels

# matplotlib.style.use("ggplot")
#matplotlib.rc('animation', html='html5')

# histograms, signals and shortcuts


def plts(signal, signal_start=0, signal_end=1e+4, offset=5):
    """Plot a signal in a give interval, control offset by hand."""
    ax1 = plt.subplot(1, 1, 1)
    ymin = np.amin(signal[signal_start:signal_end]) - offset
    ymax = np.amax(signal[signal_start:signal_end]) + offset
    ax1.set_xlim([signal_start, signal_end])
    ax1.set_ylim([ymin, ymax])
    plt.plot(signal)


def plot_signal(signal_t, signal, title="signal",
                signal_start=0, signal_end=1e+4,
                ymax=200, t_units="", units=""):
    """Given a series signal (t, signal), plot the signal."""

    ax1 = plt.subplot(1, 1, 1)
    ax1.set_xlim([signal_start, signal_end])
    ax1.set_ylim([0, ymax])
    set_plot_labels(xlabel="t ({})".format(t_units),
                  ylabel="signal ({})".format(units))
    plt.title(title)
    plt.plot(signal_t, signal)
    # plt.show()


def plot_signal_vs_time_mus(signal,
                            t_min      =    0,
                            t_max      = 1200,
                            signal_min =    0,
                            signal_max =  200,
                            figsize=(6,6)):
    """Plot signal versus time in mus (tmin, tmax in mus). """
    plt.figure(figsize=figsize)
    tstep = 25 # in ns
    PMTWL = signal.shape[0]
    signal_t = np.arange(0., PMTWL * tstep, tstep)/units.mus
    ax1 = plt.subplot(1, 1, 1)
    ax1.set_xlim([t_min, t_max])
    ax1.set_ylim([signal_min, signal_max])
    set_plot_labels(xlabel = "t (mus)",
                    ylabel = "signal (pes/adc)")
    plt.plot(signal_t, signal)


def plot_waveform(pmtwf, zoom=False, window_size=800):
    """Take as input a vector a single waveform and plot it"""

    first, last = 0, len(pmtwf)
    if zoom:
        first, last = define_window(pmtwf, window_size)

    mpl.set_plot_labels(xlabel="samples", ylabel="adc")
    plt.plot(pmtwf[first:last])


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
        set_plot_labels(xlabel="samples", ylabel="adc")
        plt.plot(wfa[i][first:last], label= 'WFA')
        plt.plot(wfb[i][first:last], label= 'WFB')
        legend = plt.legend(loc='upper right')
        for label in legend.get_texts():
            label.set_fontsize('small')


def plot_pmt_waveforms(pmtwfdf, zoom=False, window_size=800, figsize=(10,10)):
    """plot PMT wf and return figure"""
    plt.figure(figsize=figsize)
    for i in range(len(pmtwfdf)):
        first, last = 0, len(pmtwfdf[i])
        if zoom:
            first, last = define_window(pmtwfdf[i], window_size)

        ax = plt.subplot(3, 4, i+1)
        set_plot_labels(xlabel="samples", ylabel="adc")
        plt.plot(pmtwfdf[i][first:last])


def plot_pmt_signals_vs_time_mus(pmt_signals,
                                 pmt_active,
                                 t_min      =    0,
                                 t_max      = 1200,
                                 signal_min =    0,
                                 signal_max =  200,
                                 figsize=(10,10)):
    """Plot PMT signals versus time in mus  and return figure."""

    tstep = 25
    PMTWL = pmt_signals[0].shape[0]
    signal_t = np.arange(0., PMTWL * tstep, tstep)/units.mus
    plt.figure(figsize=figsize)

    for j, i in enumerate(pmt_active):
        ax1 = plt.subplot(3, 4, j+1)
        ax1.set_xlim([t_min, t_max])
        ax1.set_ylim([signal_min, signal_max])
        set_plot_labels(xlabel = "t (mus)",
                        ylabel = "signal (pes/adc)")

        plt.plot(signal_t, pmt_signals[i])


def plot_calibrated_sum_in_mus(CSUM,
                               tmin=0, tmax=1200,
                               signal_min=-5, signal_max=200,
                               csum=True, csum_mau=False):
    """Plots calibrated sums in mus (notice units)"""

    if csum:
        plot_signal_vs_time_mus(CSUM.csum,
                                t_min=tmin, t_max=tmax,
                                signal_min=signal_min, signal_max=signal_max,
                                label='CSUM')
    if csum_mau:
        plot_signal_vs_time_mus(CSUM.csum_mau,
                                t_min=tmin, t_max=tmax,
                                signal_min=signal_min, signal_max=signal_max,
                                label='CSUM_MAU')
