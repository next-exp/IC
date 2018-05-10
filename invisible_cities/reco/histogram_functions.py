from .. io   .hist_io        import save_histomanager_to_file
from .. io   .hist_io        import get_histograms_from_file
from .. evm  .histos         import Histogram
from .. evm  .histos         import HistoManager


def create_histomanager_from_dicts(histobins_dict, histolabels_dict, init_fill_dict=None):
    """
    Creates and returns an HistoManager from a dict of bins and a given of labels with identical keys.

    Arguments:
    histobins_dict   = Dictionary with keys equal to Histogram names and values equal to the binning.
    histolabels_dict = Dictionary with keys equal to Histogram names and values equal to the axis labels.
    init_fill_dict   = Dictionary with keys equal to Histogram names and values equal to an initial filling.
    """
    histo_manager = HistoManager()
    if init_fill_dict is None: init_fill_dict = dict()
    for histotitle, histobins in histobins_dict.items():
        histo_manager.new_histogram(Histogram(histotitle,
                                              histobins,
                                              histolabels_dict[histotitle],
                                              init_fill_dict.get(histotitle, None)))
    return histo_manager


def join_histograms_from_files(histofiles, group_name='HIST', join_file=None, write_mode='w'):
    """
    Joins the histograms of a given list of histogram files. If possible,
    Histograms with the same name will be added.

    histofiles = List of strings with the filenames to be summed.
    join_file  = String. If passed, saves the resulting HistoManager to this path.
    """
    if not histofiles:
        raise ValueError("List of files is empty")

    final_histogram_manager = get_histograms_from_file(histofiles[0], group_name)

    for file in histofiles[1:]:
        added_histograms        = get_histograms_from_file(file, group_name)
        final_histogram_manager = join_histo_managers     (final_histogram_manager, added_histograms)

    if join_file is not None:
        save_histomanager_to_file(final_histogram_manager, join_file, mode=write_mode, group=group_name)

    return final_histogram_manager


def join_histo_managers(histo_manager1, histo_manager2):
    """
    Joins two HistoManager. If they share histograms, the histograms are sumed.

    Arguments:
    histo_manager1, histo_manager2 = HistoManager objects to be joined.
    """
    new_histogram_manager = HistoManager()
    list_of_histograms    = set(histo_manager1.histos) | set(histo_manager2.histos)
    for histoname in list_of_histograms:
        histo1 = histo_manager1.histos.get(histoname, None)
        histo2 = histo_manager2.histos.get(histoname, None)
        try:
            new_histogram_manager.new_histogram(histo1 + histo2)
        except ValueError:
            print(f"Histograms with name {histoname} have not been added due to"
                    " incompatible binning.")
    return new_histogram_manager
