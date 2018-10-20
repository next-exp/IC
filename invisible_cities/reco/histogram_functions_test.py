import os
from   collections import defaultdict

import numpy as np

from hypothesis             import given
from hypothesis             import settings
from hypothesis.strategies  import lists

from pytest import mark

from .. reco                import histogram_functions as histf

from .. io .hist_io         import save_histomanager_to_file
from .. io .hist_io         import get_histograms_from_file
from .. evm.histos          import HistoManager
from .. evm.histos          import Histogram
from .. evm.histos_test     import assert_histogram_equality
from .. evm.histos_test     import histograms_lists
from .. evm.histos_test     import bins_arrays

@mark.skip(reason="Delaying elimination of solid cities")
@given(histograms_lists())
@settings(deadline=None)
def test_join_histo_managers(histogram_list):
    args, list_of_histograms = histogram_list
    histogram_manager        = HistoManager(list_of_histograms)
    joined_histogram_manager = histf.join_histo_managers(histogram_manager, histogram_manager)

    assert len(list_of_histograms) == len(joined_histogram_manager.histos)
    for histoname, histogram in joined_histogram_manager.histos.items():
        histo1 = histogram_manager[histoname]
        true_histogram           = Histogram(histoname, histo1.bins, histo1.labels)
        true_histogram.data      = 2 * histo1.data
        true_histogram.errors    = np.sqrt(2) * histo1.errors
        true_histogram.out_range = 2 * histo1.out_range
        assert_histogram_equality(histogram, true_histogram)


@mark.skip(reason="Delaying elimination of solid cities")
@given   (histograms_lists(), histograms_lists())
@settings(deadline=None, max_examples=700)
def test_join_histo_managers_with_different_histograms(histogram_list1, histogram_list2):
    _, list_of_histograms1   = histogram_list1
    _, list_of_histograms2   = histogram_list2
    histogram_manager1       = HistoManager(list_of_histograms1)
    histogram_manager2       = HistoManager(list_of_histograms2)
    joined_histogram_manager = histf.join_histo_managers(histogram_manager1, histogram_manager2)

    unique_histograms = set(histogram_manager1.histos) | set(histogram_manager2.histos)
    common_histograms = set(histogram_manager1.histos) & set(histogram_manager2.histos)

    remove_names      = []
    for name in unique_histograms:
        if name in common_histograms:
            if not np.all(a == b for a, b in zip(histogram_manager1[name].bins, histogram_manager2[name].bins)):
                remove_names.append(name)
    list_of_names = unique_histograms - set(remove_names)

    for histoname, histogram in joined_histogram_manager.histos.items():
        assert histoname in list_of_names
        if (histoname in histogram_manager1.histos) and (histoname in histogram_manager2.histos):
            histo1 = histogram_manager1[histoname]
            histo2 = histogram_manager2[histoname]

            true_histogram           = Histogram(histoname, histo1.bins, histo1.labels)
            true_histogram.data      =         histo1.data        + histo2.data
            true_histogram.errors    = np.sqrt(histo1.errors ** 2 + histo2.errors ** 2)
            true_histogram.out_range =         histo1.out_range   + histo2.out_range

            assert_histogram_equality(histogram, true_histogram)

        elif histoname in histogram_manager1.histos:
            histo1 = histogram_manager1[histoname]
            assert_histogram_equality(histogram, histo1)

        elif histoname in histogram_manager2.histos:
            histo2 = histogram_manager2[histoname]
            assert_histogram_equality(histogram, histo2)


@mark.skip(reason="Delaying elimination of solid cities")
@given(lists(bins_arrays(), min_size=1, max_size=5))
@settings(deadline=None)
def test_create_histomanager_from_dicts(bins):
    histobins_dict   = {}
    histolabels_dict = {}
    histograms_dict  = {}
    for i, bins_element in enumerate(bins):
        title =    f"Histo_{i}"
        labels = [ f"Xlabel_{i}", f"Ylabel_{i}" ]
        histobins_dict  [title] = bins_element
        histolabels_dict[title] = labels
        histograms_dict [title] = Histogram(title, bins_element, labels)

    histo_manager = histf.create_histomanager_from_dicts(histobins_dict, histolabels_dict)

    assert len(histograms_dict) == len(histo_manager.histos)
    for histoname, histogram in histo_manager.histos.items():
        assert_histogram_equality(histogram, histograms_dict[histoname])


@mark.skip(reason="Delaying elimination of solid cities")
@given   (histograms_lists(), histograms_lists())
@settings(deadline=None, max_examples=250)
def test_join_histograms_from_file(output_tmpdir, histogram_list1, histogram_list2):
    _, list_of_histograms1   = histogram_list1
    _, list_of_histograms2   = histogram_list2
    histogram_manager1       = HistoManager(list_of_histograms1)
    histogram_manager2       = HistoManager(list_of_histograms2)

    file_out1 = os.path.join(output_tmpdir, 'test_save_histogram_manager_1.h5')
    save_histomanager_to_file(histogram_manager1, file_out1)
    file_out2 = os.path.join(output_tmpdir, 'test_save_histogram_manager_2.h5')
    save_histomanager_to_file(histogram_manager2, file_out2)

    joined_histogram_manager1 = histf.join_histograms_from_files([file_out1, file_out2])
    joined_histogram_manager2 = histf.join_histo_managers(histogram_manager1, histogram_manager2)

    assert len(joined_histogram_manager1.histos) == len(joined_histogram_manager2.histos)
    for histoname in joined_histogram_manager1.histos:
        assert_histogram_equality(joined_histogram_manager1[histoname], joined_histogram_manager2[histoname])


@mark.skip(reason="Delaying elimination of solid cities")
@given   (histograms_lists())
@settings(deadline=None, max_examples=250)
def test_join_histograms_from_file_and_write(output_tmpdir, histogram_list):
    _, list_of_histograms   = histogram_list
    histogram_manager       = HistoManager(list_of_histograms)

    file_out_test = os.path.join(output_tmpdir , 'test_save_histogram_manager_1.h5')
    save_histomanager_to_file(histogram_manager, file_out_test)

    file_out      = os.path.join(output_tmpdir, 'test_join_histograms.h5')
    _  = histf.join_histograms_from_files([file_out_test, file_out_test], join_file=file_out)
    joined_histogram_manager1 = histf.get_histograms_from_file(file_out)
    joined_histogram_manager2 = histf.join_histo_managers(histogram_manager, histogram_manager)

    assert len(joined_histogram_manager1.histos) == len(joined_histogram_manager2.histos)
    for histoname in joined_histogram_manager1.histos:
        assert_histogram_equality(joined_histogram_manager1[histoname], joined_histogram_manager2[histoname])
