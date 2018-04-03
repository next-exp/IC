import numpy             as np
import matplotlib.pyplot as plt

from .. io   .hist_io        import get_histograms_from_file
from .. evm  .histos         import Histogram
from .. evm  .histos         import HistoManager
from .. icaro.hst_functions  import shift_to_bin_centers
from .. core .core_functions import weighted_mean_and_std


def plot_histograms_from_file(histofile, histonames='all', group_name='HIST', plot_errors=False, out_path=None):
    """
    Plots the Histograms of a given file containing an HistoManager in a 3 column plot grid.

    histofile   = String. Path to the file containing the histograms.
    histonames  = List with histogram name to be plotted, if 'all', all histograms are plotted.
    group_name  = String. Name of the group were Histograms were saved.
    plot_errors = Boolean. If true, plot the associated errors instead of the data.
    out_path    = String. Path to save the histograms in png. If not passed, histograms won't be saved.
    """
    histograms = get_histograms_from_file(histofile, group_name)
    plot_histograms(histograms, histonames=histonames, plot_errors=plot_errors, out_path=out_path)


def plot_histogram(histogram, ax=None, plot_errors=False):
    """
    Plot a Histogram.

    ax          = Axes object to plot the figure. If not passed, a new axes will be created.
    plot_errors = Boolean. If true, plot the associated errors instead of the data.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    bins        = histogram.bins
    out_range   = histogram.out_range
    labels      = histogram.labels
    title       = histogram.title
    if plot_errors:
        entries = histogram.errors
    else:
        entries = histogram.data

    if len(bins) == 1:
        ax.hist         (shift_to_bin_centers(bins[0]), bins[0],
                        weights   = entries,
                        histtype  = 'step',
                        edgecolor = 'black',
                        linewidth = 1.5)
        ax.grid         (True)
        ax.set_axisbelow(True)
        ax.set_ylabel   ("Entries", weight='bold', fontsize=20)
        out_range_string = 'Out range (%) = [{0:.2f}, {1:.2f}]'.format(get_percentage(out_range[0,0], np.sum(entries)),
                                                                       get_percentage(out_range[1,0], np.sum(entries)))

        if np.sum(entries) > 0:
            mean, std = weighted_mean_and_std(shift_to_bin_centers(bins[0]), entries)
        else:
            mean, std = 0, 0

        ax.annotate('Mean = {0:.2f}\n'.format(mean) + 'RMS = {0:.2f}\n' .format(std) +
                    out_range_string,
                    xy                  = (0.99, 0.99),
                    xycoords            = 'axes fraction',
                    fontsize            = 11,
                    weight              = 'bold',
                    horizontalalignment = 'right',
                    verticalalignment   = 'top')

    elif len(bins) == 2:
        ax.pcolormesh(bins[0], bins[1], entries.T)
        ax.set_ylabel(labels[1], weight='bold', fontsize=20)

        out_range_stringX = 'Out range X (%) = [{0:.2f}, {1:.2f}]'.format(get_percentage(out_range[0,0], np.sum(entries)),
                                                                          get_percentage(out_range[1,0], np.sum(entries)))
        out_range_stringY = 'Out range Y (%) = [{0:.2f}, {1:.2f}]'.format(get_percentage(out_range[0,1], np.sum(entries)),
                                                                          get_percentage(out_range[1,1], np.sum(entries)))

        if np.sum(entries) > 0:
            meanX, stdX = weighted_mean_and_std(shift_to_bin_centers(bins[0]), np.sum(entries, axis = 1))
            meanY, stdY = weighted_mean_and_std(shift_to_bin_centers(bins[1]), np.sum(entries, axis = 0))
        else:
            meanX, stdX = 0, 0
            meanY, stdY = 0, 0

        ax.annotate('Mean X = {0:.2f}\n'.format(meanX) + 'Mean Y = {0:.2f}\n'.format(meanY) +
                    'RMS X = {0:.2f}\n' .format(stdX)  + 'RMS Y = {0:.2f}\n' .format(stdY)  +
                    out_range_stringX + '\n' + out_range_stringY,
                    xy                  = (0.99, 0.99),
                    xycoords            = 'axes fraction',
                    fontsize            = 11,
                    weight              = 'bold',
                    color               = 'white',
                    horizontalalignment = 'right',
                    verticalalignment   = 'top')

    ax.set_xlabel      (labels[0], weight='bold', fontsize=20)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3))

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
        label.set_fontsize  (16)
    ax .xaxis.offsetText.set_fontsize  (14)
    ax .xaxis.offsetText.set_fontweight('bold')


def plot_histograms(histo_manager, histonames='all', n_columns=3, plot_errors=False, out_path=None):
    """
    Plot Histograms from a HistoManager.

    histo_manager = HistoManager object containing the Histograms to be plotted.
    histonames    = List with histogram name to be plotted, if 'all', all histograms are plotted.
    n_columns     = Int. Number of columns to distribute the histograms.
    plot_errors   = Boolean. If true, plot the associated errors instead of the data.
    out_path      = String. Path to save the histograms in png. If not passed, histograms won't be saved.
    """
    if histonames == 'all':
        histonames = histo_manager.histos

    n_histos  = len(histonames)
    n_columns = min(3, n_histos)
    n_rows    = int(np.ceil(n_histos / n_columns))

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(8 * n_columns, 6 * n_rows))

    if n_histos == 1:
        axes = np.array([axes])
    for i, histoname in enumerate(histonames):
        plot_histogram(histo_manager[histoname], ax=axes.flatten()[i], plot_errors=plot_errors)

    fig.tight_layout()

    if out_path:
        for i, histoname in enumerate(histonames):
            extent = axes.flatten()[i].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
            fig.savefig(out_path + histoname + '.png', bbox_inches=extent)

def get_percentage(a, b):
    """
    Given two flots, return the percentage between them.
    """
    return 100 * a / b if b else -100
