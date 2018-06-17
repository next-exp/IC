import numpy             as np
import matplotlib.pyplot as plt

from .. io   .hist_io        import get_histograms_from_file
from .. evm  .histos         import Histogram
from .. evm  .histos         import HistoManager
from .. icaro.hst_functions  import shift_to_bin_centers
from .. core .core_functions import weighted_mean_and_std


def plot_histograms_from_file(histofile, histonames='all', group_name='HIST', plot_errors=False, out_path=None, reference_histo=None):
    """
    Plots the Histograms of a given file containing an HistoManager in a 3 column plot grid.

    histofile       = String. Path to the file containing the histograms.
    histonames      = List with histogram name to be plotted, if 'all', all histograms are plotted.
    group_name      = String. Name of the group were Histograms were saved.
    plot_errors     = Boolean. If true, plot the associated errors instead of the data.
    out_path        = String. Path to save the histograms in png. If not passed, histograms won't be saved.
    reference_histo = String. Path to a file containing the reference histograms.
                      If not passed reference histograms won't be plotted.
    """
    histograms          = get_histograms_from_file(histofile      , group_name)
    if reference_histo:
        reference_histo = get_histograms_from_file(reference_histo, group_name)
    plot_histograms(histograms, histonames=histonames, plot_errors=plot_errors, out_path=out_path, reference_histo=reference_histo)


def plot_histogram(histogram, ax=None, plot_errors=False, draw_color='black', stats=True, normed=True):
    """
    Plot a Histogram.

    ax          = Axes object to plot the figure. If not passed, a new axes will be created.
    plot_errors = Boolean. If true, plot the associated errors instead of the data.
    draw_color  = String with the linecolor.
    stats       = Boolean. If true, histogram statistics info is added to the plotself.
    normed      = Boolean. If true, histogram is normalized.
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
                        edgecolor = draw_color,
                        linewidth = 1.5,
                        normed=normed)
        ax.grid         (True)
        ax.set_axisbelow(True)
        ax.set_ylabel   ("Entries", weight='bold', fontsize=20)

        if stats:
            entries_string   = f'Entries = {np.sum(entries):.0f}\n'
            out_range_string = 'Out range (%) = [{0:.2f}, {1:.2f}]'.format(get_percentage(out_range[0,0], np.sum(entries)),
                                                                           get_percentage(out_range[1,0], np.sum(entries)))

            if np.sum(entries) > 0:
                mean, std = weighted_mean_and_std(shift_to_bin_centers(bins[0]), entries, frequentist = True, unbiased = True)
            else:
                mean, std = 0, 0

            ax.annotate(entries_string                  +
                        'Mean = {0:.2f}\n'.format(mean) +
                        'RMS = {0:.2f}\n' .format(std)  +
                        out_range_string,
                        xy                  = (0.99, 0.99),
                        xycoords            = 'axes fraction',
                        fontsize            = 11,
                        weight              = 'bold',
                        color               = 'black',
                        horizontalalignment = 'right',
                        verticalalignment   = 'top')

    elif len(bins) == 2:
        ax.pcolormesh(bins[0], bins[1], entries.T)
        ax.set_ylabel(labels[1], weight='bold', fontsize=20)

        if stats:
            entries_string   = f'Entries = {np.sum(entries):.0f}\n'
            out_range_stringX = 'Out range X (%) = [{0:.2f}, {1:.2f}]'.format(get_percentage(out_range[0,0], np.sum(entries)),
                                                                              get_percentage(out_range[1,0], np.sum(entries)))
            out_range_stringY = 'Out range Y (%) = [{0:.2f}, {1:.2f}]'.format(get_percentage(out_range[0,1], np.sum(entries)),
                                                                              get_percentage(out_range[1,1], np.sum(entries)))

            if np.sum(entries) > 0:
                meanX, stdX = weighted_mean_and_std(shift_to_bin_centers(bins[0]), np.sum(entries, axis = 1), frequentist = True, unbiased = True)
                meanY, stdY = weighted_mean_and_std(shift_to_bin_centers(bins[1]), np.sum(entries, axis = 0), frequentist = True, unbiased = True)
            else:
                meanX, stdX = 0, 0
                meanY, stdY = 0, 0

            ax.annotate(entries_string +
                        'Mean X = {0:.2f}\n'.format(meanX) + 'Mean Y = {0:.2f}\n'.format(meanY) +
                        'RMS X = {0:.2f}\n' .format(stdX)  + 'RMS Y = {0:.2f}\n' .format(stdY)  +
                        out_range_stringX + '\n' + out_range_stringY,
                        xy                  = (0.99, 0.99),
                        xycoords            = 'axes fraction',
                        fontsize            = 11,
                        weight              = 'bold',
                        color               = 'white',
                        horizontalalignment = 'right',
                        verticalalignment   = 'top')

    elif len(bins) == 3:
        ave = np   .apply_along_axis(average_empty, 2, entries, shift_to_bin_centers(bins[2]))
        ave = np.ma.masked_array    (ave, ave < 0.00001)

        img = ax .pcolormesh      (bins[0], bins[1], ave.T)
        cb  = plt.colorbar        (img, ax=ax)
        cb       .set_label       (labels[2], weight='bold', fontsize=20)
        for label in cb.ax.yaxis.get_ticklabels():
            label.set_weight      ("bold")
            label.set_fontsize    (16)

        ax.set_ylabel(labels[1], weight='bold', fontsize=20)

    ax.set_xlabel      (labels[0], weight='bold', fontsize=20)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3))

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
        label.set_fontsize  (16)
    ax .xaxis.offsetText.set_fontsize  (14)
    ax .xaxis.offsetText.set_fontweight('bold')


def plot_histograms(histo_manager, histonames='all', n_columns=3, plot_errors=False, out_path=None, reference_histo=None, normed=True):
    """
    Plot Histograms from a HistoManager.

    histo_manager   = HistoManager object containing the Histograms to be plotted.
    histonames      = List with histogram name to be plotted, if 'all', all histograms are plotted.
    n_columns       = Int. Number of columns to distribute the histograms.
    plot_errors     = Boolean. If true, plot the associated errors instead of the data.
    out_path        = String. Path to save the histograms in png. If not passed, histograms won't be saved.
    reference_histo = HistoManager object containing the Histograms to be plotted as reference.
    normed          = Boolean. If true, histograms are normalized.
    """
    if histonames == 'all':
        histonames = histo_manager.histos

    if out_path is None:
        n_histos  = len(histonames)
        n_columns = min(3, n_histos)
        n_rows    = int(np.ceil(n_histos / n_columns))

        fig, axes = plt.subplots(n_rows, n_columns, figsize=(8 * n_columns, 6 * n_rows))

    for i, histoname in enumerate(histonames):
        if out_path:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        else:
            ax = axes.flatten()[i] if isinstance(axes, np.ndarray) else axes
        if reference_histo:
            if len(reference_histo[histoname].bins) == 1:
                plot_histogram(reference_histo[histoname], ax=ax, plot_errors=plot_errors, normed=normed, draw_color='red', stats=False)
        plot_histogram        (histo_manager  [histoname], ax=ax, plot_errors=plot_errors, normed=normed)

        if out_path:
            fig.tight_layout()
            fig.savefig(out_path + histoname + '.png')
            fig.clf()
            plt.close(fig)
    if out_path is None:
        fig.tight_layout()


def get_percentage(a, b):
    """
    Given two flots, return the percentage between them.
    """
    return 100 * a / b if b else -100


def average_empty(x, bins):
    """
    Returns the weighted mean. If all weights are 0, the mean is considered to be 0.
    """
    return np.average(bins, weights=x) if np.any(x > 0.) else 0.
