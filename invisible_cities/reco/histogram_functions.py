import numpy as np
import tables as tb
import matplotlib.pyplot as plt

from .. evm.histos import Histogram
from .. evm.histos import HistoManager

from .. core.core_functions import weighted_mean_and_std
from .. icaro.hst_functions import shift_to_bin_centers

def create_histomanager_from_dicts(histobins_dict, histolabels_dict, init_fill_dict={}):
    histo_manager = HistoManager()
    for histotitle, histobins in histobins_dict.items():
        histo_manager.new_histogram(Histogram(histotitle, histobins, histolabels_dict[histotitle],
                                    init_fill_dict.get(histotitle, None)))
    return histo_manager


def get_histograms_from_file(file_input):
    histo_manager = HistoManager()
    with tb.open_file(file_input, "r") as h5in:

        name_sel = lambda x: (    ('bins'     not in x)
                              and ('labels'   not in x)
                              and ('errors'   not in x)
                              and ('outRange' not in x))

        histogram_list = []

        for histoname in filter(name_sel, list(h5in.root.HIST._v_children)):
            entries   = np.array(getattr(h5in.root.HIST, histoname)[:])
            bins      =          getattr(h5in.root.HIST, histoname + '_bins')[:]
            out_range =          getattr(h5in.root.HIST, histoname + '_outRange')[:]
            errors    = np.array(getattr(h5in.root.HIST, histoname + '_errors')[:])
            labels    =          getattr(h5in.root.HIST, histoname + '_labels')[:]
            labels    = [str(lab)[2:-1].replace('\\\\', '\\') for lab in labels]

            histogram = Histogram(histoname, bins, labels)

            histogram.data      = entries
            histogram.out_range = out_range
            histogram.errors    = errors

            histogram_list.append(histogram)

    return HistoManager(histogram_list)


def join_histograms_from_files(histofiles, join_file=None):
    """
    Joins the histograms of a given list of histogram files. If possible,
    Histograms with the same name will be added.

    histofiles = List of strings with the filenames to be summed.
    join_file  = String. If passed, saves the resulting HistoManager to this path.
    """

    if len(histofiles)<1:
        raise ValueError("List of files is empty")

    final_histogram_manager = get_histograms_from_file(histofiles[0])

    for file in histofiles[1:]:
        added_histograms = get_histograms_from_file(file)
        final_histogram_manager = join_histo_managers(final_histogram_manager, added_histograms)

    if join_file is not None:
        final_histogram_manager.save_to_file(join_file)

    return final_histogram_manager


def plot_histograms_from_file(histofile, histonames='all', plot_errors=False, out_path=None):
    """
    Plots the histograms of a given histogram file in a 3 column plot grid.

    histofile   = File containing the histograms.
    histonames  = List with histogram name to be plotted, if 'all', all histograms are plotted.
    plot_errors = Boolean. If true, plot the associated errors instead of the data.
    save_pdf    = Length 2 list, first element is a boolean. If true, saves the histograms
                  separately in pdf on the path passed as second element of the list.
    """
    histograms = get_histograms_from_file(histofile)
    plot_histograms(histograms, histonames=histonames, plot_errors=plot_errors, out_path=out_path)


def plot_histogram(histogram, ax=None, plot_errors=False):
    if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    bins      = histogram.bins
    out_range = histogram.out_range
    labels    = histogram.labels
    title     = histogram.title
    if plot_errors:
        entries = histogram.errors
    else:
        entries = histogram.data

    if len(bins) == 1:
        ax.bar(shift_to_bin_centers(bins[0]), entries, width=np.diff(bins[0]))
        ax.set_ylabel("Entries", weight='bold', fontsize=20)
        out_range_string = 'Out range (%) = [{0:.2f}, {1:.2f}]'.format(get_percentage(out_range[0,0], np.sum(entries)),
                                                                       get_percentage(out_range[1,0], np.sum(entries)))

        mean, std = weighted_mean_and_std(shift_to_bin_centers(bins[0]), entries)

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

        meanX, stdX = weighted_mean_and_std(shift_to_bin_centers(bins[0]), np.sum(entries, axis = 1))
        meanY, stdY = weighted_mean_and_std(shift_to_bin_centers(bins[1]), np.sum(entries, axis = 0))

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

    ax.set_xlabel(labels[0], weight='bold', fontsize=20)
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3))

    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontweight('bold')
        label.set_fontsize(16)
    ax.xaxis.offsetText.set_fontsize(14)
    ax.xaxis.offsetText.set_fontweight('bold')


def plot_histograms(histo_manager, histonames='all', n_columns=3, plot_errors=False, out_path=None):
    if histonames == 'all':
            histonames = histo_manager#.histos

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


def join_histo_managers(histo_manager1, histo_manager2):
    """
    Joins two HistoManager. If they share histograms, the histograms are sumed.

    Arguments:
    histo_manager1, histo_manager2 = HistoManager objects to be joined.
    """
    new_histogram_manager = HistoManager()
    list_of_histograms = set(histo_manager1.histos) | set(histo_manager2.histos)
    for histoname in list_of_histograms:
        histo1 = histo_manager1.histos.get(histoname, None)
        histo2 = histo_manager2.histos.get(histoname, None)
        try:
            new_histogram_manager.new_histogram(histo1 + histo2)
        except ValueError:
            print(f"Histograms with name {histoname} have not been added due to"
                    " incompatible binning.")
    return new_histogram_manager


def get_percentage(a, b):
    if b == 0:
        return -100.
    return 100. * a / b
