import os
from   collections import defaultdict

import numpy as np

from hypothesis             import given
from hypothesis             import settings
from hypothesis             import HealthCheck
from hypothesis.strategies  import lists

from .. reco                import histogram_functions as histf

from .. evm.histos          import HistoManager
from .. evm.histos          import Histogram
from .. evm.histos_test     import assert_histogram_equality
from .. evm.histos_test     import histograms_lists
from .. evm.histos_test     import bins_arrays


@given(histograms_lists())
def test_join_histo_managers(histogram_list):
    args, list_of_histograms = histogram_list
    histogram_manager        = HistoManager(list_of_histograms)
    joined_histogram_manager = histf.join_histo_managers(histogram_manager, histogram_manager)

    assert len(list_of_histograms) == len(joined_histogram_manager.histos)
    for histoname, histogram in joined_histogram_manager.histos.items():
        histo1 = histogram_manager[histoname]
        aux_histogram           = Histogram(histoname, histo1.bins, histo1.labels)
        aux_histogram.data      = 2 * histo1.data
        aux_histogram.errors    = np.sqrt(2) * histo1.errors
        aux_histogram.out_range = 2 * histo1.out_range
        assert_histogram_equality(histogram, aux_histogram)


@settings(suppress_health_check=(HealthCheck.too_slow, HealthCheck.filter_too_much,))
@given   (histograms_lists(), histograms_lists())
def test_join_histo_managers_with_different_histograms(histogram_list1, histogram_list2):
    _, list_of_histograms1   = histogram_list1
    _, list_of_histograms2   = histogram_list2
    histogram_manager1       = HistoManager(list_of_histograms1)
    histogram_manager2       = HistoManager(list_of_histograms2)
    joined_histogram_manager = histf.join_histo_managers(histogram_manager1, histogram_manager2)

    list_of_names = set(histogram_manager1.histos) | set(histogram_manager2.histos)
    assert len(list_of_names) == len(joined_histogram_manager.histos)

    for histoname, histogram in joined_histogram_manager.histos.items():
        assert histoname in list_of_names
        if (histoname in histogram_manager1.histos) and (histoname in histogram_manager2.histos):
            histo1 = histogram_manager1[histoname]
            histo2 = histogram_manager2[histoname]

            aux_histogram           = Histogram(histoname, histo1.bins, histo1.labels)
            aux_histogram.data      =         histo1.data        + histo2.data
            aux_histogram.errors    = np.sqrt(histo1.errors ** 2 + histo2.errors ** 2)
            aux_histogram.out_range =         histo1.out_range   + histo2.out_range

            assert_histogram_equality(histogram, aux_histogram)

        elif histoname in histogram_manager1.histos:
            histo1 = histogram_manager1[histoname]
            assert_histogram_equality(histogram, histo1)

        elif histoname in histogram_manager2.histos:
            histo2 = histogram_manager2[histoname]
            assert_histogram_equality(histogram, histo2)


@given(lists(bins_arrays(), min_size=1, max_size=5))
def test_create_histomanager_from_dicts(bins):
    histobins_dict   = {}
    histolabels_dict = {}
    histograms_dict  = {}
    print(bins)
    for i, bins_element in enumerate(bins):
        print(bins_element)
        title =    "Histo_{}" .format(i)
        labels = [ "Xlabel_{}".format(i), "Ylabel_{}".format(i) ]
        histobins_dict  [title] = bins_element
        histolabels_dict[title] = labels
        histograms_dict [title] = Histogram(title, bins_element, labels)

    histo_manager = histf.create_histomanager_from_dicts(histobins_dict, histolabels_dict)

    assert len(histograms_dict) == len(histo_manager.histos)
    for histoname, histogram in histo_manager.histos.items():
        assert_histogram_equality(histogram, histograms_dict[histoname])


@settings(suppress_health_check=(HealthCheck.too_slow, HealthCheck.filter_too_much,))
@given(histograms_lists())
def test_get_histograms_from_file(output_tmpdir, histogram_list):
    args, list_of_histograms  = histogram_list
    histogram_manager1        = HistoManager(list_of_histograms)

    file_out = os.path.join(output_tmpdir, 'test_save_histogram_manager.h5')
    histogram_manager1.save_to_file(file_out)

    histogram_manager2 = histf.get_histograms_from_file(file_out)

    assert len(histogram_manager1.histos) == len(histogram_manager2.histos)
    for histoname in histogram_manager1.histos:
        assert_histogram_equality(histogram_manager1[histoname], histogram_manager2[histoname])


@settings(suppress_health_check=(HealthCheck.too_slow, HealthCheck.filter_too_much,))
@given   (histograms_lists(), histograms_lists())
def test_join_histograms_from_file(output_tmpdir, histogram_list1, histogram_list2):
    _, list_of_histograms1   = histogram_list1
    _, list_of_histograms2   = histogram_list2
    histogram_manager1       = HistoManager(list_of_histograms1)
    histogram_manager2       = HistoManager(list_of_histograms2)

    file_out1 = os.path.join(output_tmpdir, 'test_save_histogram_manager_1.h5')
    histogram_manager1.save_to_file(file_out1)
    file_out2 = os.path.join(output_tmpdir, 'test_save_histogram_manager_2.h5')
    histogram_manager2.save_to_file(file_out2)

    joined_histogram_manager1 = histf.join_histograms_from_files([file_out1, file_out2])
    joined_histogram_manager2 = histf.join_histo_managers(histogram_manager1, histogram_manager2)

    assert len(joined_histogram_manager1.histos) == len(joined_histogram_manager2.histos)
    for histoname in joined_histogram_manager1.histos:
        assert_histogram_equality(joined_histogram_manager1[histoname], joined_histogram_manager2[histoname])

@settings(suppress_health_check=(HealthCheck.too_slow, HealthCheck.filter_too_much,))
@given   (histograms_lists())
def test_join_histograms_from_file_and_write(output_tmpdir, histogram_list):
    _, list_of_histograms   = histogram_list
    histogram_manager       = HistoManager(list_of_histograms)

    file_out_aux = os.path.join(output_tmpdir, 'test_save_histogram_manager_1.h5')
    histogram_manager.save_to_file(file_out_aux)

    file_out     = os.path.join(output_tmpdir, 'test_join_histograms.h5')
    _  = histf.join_histograms_from_files([file_out_aux, file_out_aux], join_file=file_out)
    joined_histogram_manager1 = histf.get_histograms_from_file(file_out)
    joined_histogram_manager2 = histf.join_histo_managers(histogram_manager, histogram_manager)

    assert len(joined_histogram_manager1.histos) == len(joined_histogram_manager2.histos)
    for histoname in joined_histogram_manager1.histos:
        assert_histogram_equality(joined_histogram_manager1[histoname], joined_histogram_manager2[histoname])
