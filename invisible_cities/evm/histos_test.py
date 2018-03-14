import numpy  as np
import tables as tb
import os
import string

from invisible_cities.evm.histos import HistoManager, Histogram

from pytest import raises

from hypothesis             import assume
from hypothesis             import given
from hypothesis             import settings
from hypothesis             import HealthCheck
from hypothesis.strategies  import composite
from hypothesis.extra.numpy import arrays
from hypothesis.strategies  import integers
from hypothesis.strategies  import floats
from hypothesis.strategies  import text
from hypothesis.strategies  import lists
from hypothesis.strategies  import sampled_from
from hypothesis.strategies  import one_of

characters = tuple(string.ascii_letters + string.digits + "-")

def assert_histogram_equality(histogram1, histogram2):
    assert np.all     (a == b for a, b in zip(histogram1.bins, histogram2.bins))
    assert np.allclose(     histogram1.data,      histogram2.data)
    assert np.allclose(   histogram1.errors,    histogram2.errors)
    assert np.allclose(histogram1.out_range, histogram2.out_range)
    assert histogram1.title  == histogram2.title
    assert histogram1.labels == histogram2.labels

@composite
def bins_arrays(draw, dimension=0):
    if dimension <= 0:
        dimension = draw(sampled_from((1,2)))

    bin_margins = draw(arrays(float, (dimension, 2),
                        floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)))

    for bin_margin in bin_margins:
        assume(round(bin_margin.min(), 2) < round(bin_margin.max(), 2))

    bins = [ np.linspace(bin_margin.min(), bin_margin.max(), draw(integers(2, 20))) for bin_margin in bin_margins ]

    return bins


@composite
def filled_histograms(draw, dimension=0, fixed_bins=None):
    if fixed_bins is not None:
        dimension = len(fixed_bins)
    if dimension <= 0:
        dimension = draw(sampled_from((1,2)))

    if fixed_bins is not None:
        bins = [ np.linspace(bin_range[0], bin_range[1], bin_range[2] + 1) for bin_range in fixed_bins ]
    else:
        bins = draw(bins_arrays(dimension=dimension))

    title  = draw(text(characters, min_size=1))
    labels = draw(lists(text(characters, min_size=1), min_size=dimension, max_size=dimension))
    shape  = draw(integers(50, 100)),
    data   = []
    for i in range(dimension):
        lower_limit = bins[i][0]  * draw(floats(0.5, 1.5))
        upper_limit = bins[i][-1] * draw(floats(0.5, 1.5))
        assume(lower_limit < upper_limit)
        data.append(draw(arrays(float, shape, floats(lower_limit, upper_limit,
                                                     allow_nan=False, allow_infinity=False))))
    data = np.array(data)
    args = title, bins, labels, data
    return args, Histogram(title, bins, labels, data)


@composite
def empty_histograms(draw, dimension=0):
    if dimension <= 0:
        dimension = draw(sampled_from((1,2)))
    bins   = draw(bins_arrays(dimension=dimension))
    title  = draw(text(characters, min_size=1))
    labels = draw(lists(text(characters, min_size=1), min_size=dimension, max_size=dimension))
    args   = title, bins, labels
    return args, Histogram(title, bins, labels)


@composite
def histograms_lists(draw, number=0, dimension=0, fixed_bins=None):
    if number <= 0:
        number = draw(integers(2, 5))
    empty_histogram  =  empty_histograms(dimension=dimension)
    filled_histogram = filled_histograms(dimension=dimension, fixed_bins=fixed_bins)
    args, histograms = zip(*[ draw(one_of(empty_histogram, filled_histogram)) for i in range(number) ])

    titles, *_ = zip(*args)
    assume(len(set(titles)) == len(titles))

    return args, histograms


@given(bins_arrays())
def test_histogram_initialization(bins):
    label      = [ 'Random distribution' ]
    title      = 'Test_histogram'
    test_histo = Histogram(title, bins, label)

    assert np.all     (a == b for a, b in zip(test_histo.bins, bins))
    assert np.allclose(     test_histo.data, np.zeros(shape=tuple(len(x) - 1 for x in bins)))
    assert np.allclose(test_histo.out_range,                  np.zeros(shape=(2, len(bins))))
    assert np.allclose(   test_histo.errors, np.zeros(shape=tuple(len(x) - 1 for x in bins)))
    assert test_histo.labels == label
    assert test_histo.title  == title


@given(bins_arrays())
def test_histogram_initialization_with_values(bins):
    label     = [ 'Random distribution' ]
    title     = 'Test_histogram'
    data      = []
    out_range = []
    for ibin in bins:
        lower_limit = ibin[0]  * 0.95
        upper_limit = ibin[-1] * 1.05
        data.append(np.random.uniform(lower_limit, upper_limit, 500))
        out_range.append(np.array([ np.count_nonzero(data[-1] < ibin[0]),
                                    np.count_nonzero(data[-1] > ibin[-1])]))
    data         = np.array(data)
    test_histo   = Histogram(title, bins, label, data)
    out_range    = np.array(out_range).T
    binned_data  = np.histogramdd(data.T, bins)[0]

    assert np.all     (a == b for a, b in zip(test_histo.bins, bins))
    assert np.allclose(     test_histo.data,             binned_data)
    assert np.allclose(test_histo.out_range,               out_range)
    assert np.allclose(   test_histo.errors,    np.sqrt(binned_data))
    assert test_histo.labels == label
    assert test_histo.title  == title


@given(empty_histograms())
def test_histogram_fill(empty_histogram):
    _, test_histogram = empty_histogram
    histobins         = test_histogram.bins
    n_points          = np.random.random_integers(5, 200, 1)
    test_data         = [ np.random.uniform( bins[0] * (1 + np.random.uniform()),
                                             bins[-1] * (1 + np.random.uniform()),
                                             n_points) for bins in histobins ]

    test_histogram.fill(test_data)
    out_of_range = [ np.array([ np.count_nonzero(test_data[i] < bins[0]),
                                np.count_nonzero(test_data[i] > bins[-1])])
                                for i, bins in enumerate(histobins) ]
    test_out_of_range  = np.array(out_of_range).T
    test_data          = np.histogramdd(np.asarray(test_data).T, test_histogram.bins)[0]
    test_errors        = np.sqrt(test_data)

    assert np.allclose(     test_histogram.data,         test_data)
    assert np.allclose(   test_histogram.errors,       test_errors)
    assert np.allclose(test_histogram.out_range, test_out_of_range)


@given(empty_histograms())
def test_histogram_fill_with_weights(empty_histogram):
    _, test_histogram = empty_histogram
    histobins         = test_histogram.bins
    n_points          = np.random.random_integers(50, 200, 1)
    test_data         = [ np.random.uniform( bins[0] * (1 + np.random.uniform()),
                                             bins[-1] * (1 + np.random.uniform()),
                                             n_points) for bins in histobins ]
    test_weights = np.random.uniform(1, 10, n_points)
    test_histogram.fill(test_data, test_weights)

    out_of_range = [ np.array([ np.count_nonzero(test_data[i] < bins[0]),
                                np.count_nonzero(test_data[i] > bins[-1])])
                                for i, bins in enumerate(histobins) ]
    test_out_of_range  = np.array(out_of_range).T
    test_data          = np.histogramdd(np.asarray(test_data).T, test_histogram.bins, weights=test_weights)[0]
    test_errors        = np.sqrt(test_data)

    assert np.allclose(     test_histogram.data,         test_data)
    assert np.allclose(   test_histogram.errors,       test_errors)
    assert np.allclose(test_histogram.out_range, test_out_of_range)


def test_bin_data():
    bins       = [ np.linspace(0., 5., 6) ]
    label      = [ 'Random distribution' ]
    title      = 'Test_histogram'
    test_histo = Histogram(title, bins, label)

    data           = np.array([-1., 2.2, 3.2, 4.5, 6.3, 7.1, 4.9, 3.1, 0.2, 2.1, 2.2, 1.1])
    test_data      = [1, 1, 3, 2, 2]
    test_out_range = [[1], [2]]

    binned_data, out_range = test_histo.bin_data(data, data_weights=np.ones(len(data)))

    assert np.allclose(binned_data,      test_data)
    assert np.allclose(  out_range, test_out_range)


def test_count_out_of_range():
    bins       = [ np.linspace(0., 5., 6) ]
    label      = [ 'Random distribution' ]
    title      = 'Test_histogram'
    test_histo = Histogram(title, bins, label)

    data           = np.array([-1., 2.2, 3.2, 4.5, 6.3, 7.1, 4.9, 3.1, 0.2, 2.1, 2.2, 1.1])
    test_data      = [1, 1, 3, 2, 2]
    test_out_range = [[1], [2]]

    out_range = test_histo.count_out_of_range(np.array(data, ndmin=2))

    assert np.allclose(out_range, test_out_range)


@given(filled_histograms())
def test_update_errors(filled_histogram):
    _, test_histogram     = filled_histogram
    test_histogram.errors = None
    test_histogram.update_errors()
    assert np.allclose(test_histogram.errors, np.sqrt(test_histogram.data))


@given(filled_histograms())
def test_update_errors_with_values(filled_histogram):
    _, test_histogram = filled_histogram
    new_errors        = np.random.uniform(0., 1000, size=test_histogram.data.shape)
    test_histogram.update_errors(new_errors)
    assert np.allclose(test_histogram.errors, new_errors)


@given(filled_histograms(fixed_bins=[[50, 900, 20.]]),
       filled_histograms(fixed_bins=[[50, 900, 20.]]))
def test_add_histograms(first_histogram, second_histogram):
    _, test_histogram1 = first_histogram
    _, test_histogram2 = second_histogram
    sum_histogram      = test_histogram1 + test_histogram2

    assert np.all     (a == b for a, b in zip(sum_histogram.bins, test_histogram1.bins))
    assert np.all     (a == b for a, b in zip(sum_histogram.bins, test_histogram2.bins))
    assert np.allclose(     sum_histogram.data,         test_histogram1.data        + test_histogram2.data        )
    assert np.allclose(   sum_histogram.errors, np.sqrt(test_histogram1.errors ** 2 + test_histogram2.errors ** 2))
    assert np.allclose(sum_histogram.out_range,         test_histogram1.out_range   + test_histogram2.out_range   )
    assert sum_histogram.labels == test_histogram1.labels
    assert sum_histogram.title  == test_histogram1.title


@given(filled_histograms(fixed_bins=[[50, 900, 20.]]),
       filled_histograms(fixed_bins=[[50, 900, 5.]]))
def test_add_histograms_with_incompatible_binning_raises_ValueError(first_histogram, second_histogram):
    _, test_histogram1 = first_histogram
    _, test_histogram2 = second_histogram

    with raises(ValueError):
        sum_histogram = test_histogram1 + test_histogram2


@given(filled_histograms(fixed_bins=[[50, 900, 20.], [20, 180, 15]]),
       filled_histograms(fixed_bins=[[50, 900, 20.], [20, 180, 15]]))
def test_add_histograms_2d(first_histogram, second_histogram):
    _, test_histogram1 = first_histogram
    _, test_histogram2 = second_histogram
    sum_histogram      = test_histogram1 + test_histogram2

    assert np.all     (a == b for a, b in zip(sum_histogram.bins, test_histogram1.bins))
    assert np.all     (a == b for a, b in zip(sum_histogram.bins, test_histogram2.bins))
    assert np.allclose(     sum_histogram.data,         test_histogram1.data        + test_histogram2.data        )
    assert np.allclose(   sum_histogram.errors, np.sqrt(test_histogram1.errors ** 2 + test_histogram2.errors ** 2))
    assert np.allclose(sum_histogram.out_range,         test_histogram1.out_range   + test_histogram2.out_range   )
    assert sum_histogram.labels == test_histogram1.labels
    assert sum_histogram.title  == test_histogram1.title


@settings(suppress_health_check=(HealthCheck.too_slow,))
@given(histograms_lists())
def test_histomanager_initialization_with_histograms(histogram_list):
    _, list_of_histograms = histogram_list
    histogram_manager     = HistoManager(list_of_histograms)

    for histogram in list_of_histograms:
        assert_histogram_equality(histogram, histogram_manager[histogram.title])


def test_histomanager_initialization_without_histograms():
    histogram_manager = HistoManager()
    assert len(histogram_manager.histos) == 0


@given(one_of(empty_histograms(), filled_histograms()))
def test_new_histogram_in_histomanager(test_histogram):
    _, histogram      = test_histogram
    histoname         = histogram.title
    histogram_manager = HistoManager()
    histogram_manager.new_histogram(histogram)

    assert_histogram_equality(histogram, histogram_manager[histoname])


@settings(suppress_health_check=(HealthCheck.too_slow,))
@given(histograms_lists())
def test_fill_histograms_in_histomanager(histogram_list):
    args, list_of_histograms = histogram_list
    titles, histobins, *_    = zip(*args)
    histogram_manager        = HistoManager(list_of_histograms)

    test_data_values  = {}
    old_data_values   = {}
    test_out_of_range = {}
    old_out_of_range  = {}
    for i, title in enumerate(titles):
        old_data_values[title]  = np.copy(histogram_manager[title].data)
        old_out_of_range[title] = np.copy(histogram_manager[title].out_range)
        n_points                = np.random.random_integers(5, 200, 1)
        test_data               = [ np.random.uniform( bins[0] * (1 + np.random.uniform()),
                                    bins[-1] * (1 + np.random.uniform()),
                                    n_points) for bins in histobins[i] ]
        test_data_values[title]  = test_data
        out_of_range             = [ np.array([ np.count_nonzero(test_data[j] < bins[0]),
                                                np.count_nonzero(test_data[j] > bins[-1])])
                                    for j, bins in enumerate(histobins[i]) ]
        test_out_of_range[title] = np.array(out_of_range).T

    histogram_manager.fill_histograms(test_data_values)

    for histoname, data in test_data_values.items():
        histogram = histogram_manager[histoname]
        old_data  = np.array(old_data_values[histoname])
        test_data = np.histogramdd(np.asarray(data).T, histogram.bins)[0]
        assert np.allclose(     histogram.data,                                      test_data + old_data )
        assert np.allclose(   histogram.errors,                              np.sqrt(test_data + old_data))
        assert np.allclose(histogram.out_range, test_out_of_range[histoname] + old_out_of_range[histoname])


@settings(suppress_health_check=(HealthCheck.too_slow,))
@given(histograms_lists(), text(characters, min_size=1))
def test_save_to_file_write_mode(output_tmpdir, histogram_list, group):
    args, list_of_histograms = histogram_list
    histogram_manager        = HistoManager(list_of_histograms)

    file_out = os.path.join(output_tmpdir, 'test_save_histogram_manager.h5')
    histogram_manager.save_to_file(file_out, group=group)

    with tb.open_file(file_out, "r") as h5in:
        file_group = getattr(h5in.root, group)
        for histogram in list_of_histograms:
            histoname = histogram.title
            saved_labels = [str(label)[2:-1].replace('\\\\', '\\') for label in getattr(file_group, histoname + "_labels")[:]]

            assert histoname in file_group
            assert np.all     (a == b for a, b in zip(histogram.bins, getattr(file_group, histoname + "_bins")[:]))
            assert np.allclose(     histogram.data, getattr(file_group, histoname              )[:])
            assert np.allclose(   histogram.errors, getattr(file_group, histoname + "_errors"  )[:])
            assert np.allclose(histogram.out_range, getattr(file_group, histoname + "_outRange")[:])
            assert histogram.labels == saved_labels


@settings(suppress_health_check=(HealthCheck.too_slow,))
@given(histograms_lists(), text(characters, min_size=1))
def test_save_to_file_append_mode(output_tmpdir, histogram_list, group):
    args, list_of_histograms = histogram_list
    histogram_manager        = HistoManager(list_of_histograms[:1])

    file_out = os.path.join(output_tmpdir, 'test_save_histogram_manager.h5')
    histogram_manager.save_to_file(file_out, mode='w', group=group)

    histogram_manager = HistoManager(list_of_histograms[1:])
    histogram_manager.save_to_file(file_out, mode='a', group=group)

    with tb.open_file(file_out, "r") as h5in:
        file_group = getattr(h5in.root, group)
        for histogram in list_of_histograms:
            histoname    = histogram.title
            saved_labels = [str(label)[2:-1].replace('\\\\', '\\') for label in getattr(file_group, histoname + "_labels")[:]]

            assert histoname in file_group
            assert np.all     (a == b for a, b in zip(histogram.bins, getattr(file_group, histoname + "_bins")[:]))
            assert np.allclose(     histogram.data, getattr(file_group, histoname              )[:])
            assert np.allclose(   histogram.errors, getattr(file_group, histoname + "_errors"  )[:])
            assert np.allclose(histogram.out_range, getattr(file_group, histoname + "_outRange")[:])
            assert histogram.labels == saved_labels


@settings(suppress_health_check=(HealthCheck.too_slow,))
@given(histograms_lists(), text(characters, min_size=1), text(characters, min_size=1, max_size=1))
def test_save_to_file_raises_ValueError(output_tmpdir, histogram_list, group, write_mode):
    args, list_of_histograms = histogram_list
    histogram_manager        = HistoManager(list_of_histograms)

    file_out = os.path.join(output_tmpdir, 'test_save_histogram_manager.h5')
    write_mode = write_mode.replace('w','')
    write_mode = write_mode.replace('a','')
    with raises(ValueError):
        histogram_manager.save_to_file(file_out, mode=write_mode, group=group)


@settings(suppress_health_check=(HealthCheck.too_slow,))
@given(histograms_lists())
def test_getitem_histomanager(histogram_list):
    args, list_of_histograms = histogram_list
    titles, histobins, *_ = zip(*args)
    histogram_manager = HistoManager(list_of_histograms)

    for histoname in histogram_manager.histos:
        assert np.all     (a == b for a, b in zip(histogram_manager.histos[histoname].bins, histogram_manager[histoname].bins))
        assert np.allclose(histogram_manager.histos[histoname].     data, histogram_manager[histoname].     data)
        assert np.allclose(histogram_manager.histos[histoname].   errors, histogram_manager[histoname].   errors)
        assert np.allclose(histogram_manager.histos[histoname].out_range, histogram_manager[histoname].out_range)
        assert histogram_manager.histos[histoname].labels == histogram_manager[histoname].labels
        assert histogram_manager.histos[histoname].title  == histogram_manager[histoname].title


@settings(suppress_health_check=(HealthCheck.too_slow,))
@given(histograms_lists())
def test_setitem_histomanager(histogram_list):
    args, list_of_histograms = histogram_list
    titles, histobins, *_    = zip(*args)
    histogram_manager1       = HistoManager()
    histogram_manager2       = HistoManager()

    for histogram in list_of_histograms:
        histoname                            = histogram.title
        histogram_manager1.histos[histoname] = histogram
        histogram_manager2[histoname]        = histogram

        assert_histogram_equality(histogram_manager1[histoname], histogram_manager2[histoname])
