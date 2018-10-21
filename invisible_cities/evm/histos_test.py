import string

import numpy  as np

from pytest import raises
from pytest import mark

from hypothesis             import assume
from hypothesis             import given
from hypothesis             import settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies  import composite
from hypothesis.strategies  import integers
from hypothesis.strategies  import floats
from hypothesis.strategies  import text
from hypothesis.strategies  import lists
from hypothesis.strategies  import sampled_from
from hypothesis.strategies  import one_of

from .. evm.histos  import HistoManager, Histogram


characters = string.ascii_letters + string.digits
letters    = string.ascii_letters


def assert_histogram_equality(histogram1, histogram2):
    assert np.all     (a == b for a, b in zip(histogram1.bins, histogram2.bins))
    assert np.allclose(histogram1.data     , histogram2.data     )
    assert np.allclose(histogram1.errors   , histogram2.errors   )
    assert np.allclose(histogram1.out_range, histogram2.out_range)
    assert             histogram1.title   == histogram2.title
    assert             histogram1.labels  == histogram2.labels


@composite
def titles(draw):
    return draw(text(letters, min_size=1)) + draw(text(characters, min_size=5))


@composite
def bins_arrays(draw, dimension=0):
    if dimension <= 0:
        dimension = draw(sampled_from((1,2)))

    bin_lower_margins = draw(arrays(float, dimension,
                             floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)))
    bin_additive      = draw(arrays(float, dimension,
                             floats(1.1 , 1e3, allow_nan=False, allow_infinity=False)))
    bin_upper_margins = bin_lower_margins + bin_additive

    bins = [ np.linspace(bin_lower_margins[i], bin_upper_margins[i], draw(integers(2, 20))) for i, _ in enumerate(bin_lower_margins) ]

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

    labels = draw(lists(text(characters, min_size=5), min_size=dimension, max_size=dimension))
    shape  = draw(integers(50, 100)),
    data   = []
    for i in range(dimension):
        lower_limit = bins[i][0]  - draw(floats(0.5, 1e8, allow_nan=False, allow_infinity=False))
        upper_limit = bins[i][-1] + draw(floats(0.5, 1e8, allow_nan=False, allow_infinity=False))
        data.append(draw(arrays(float, shape, floats(lower_limit, upper_limit,
                                                     allow_nan=False, allow_infinity=False))))
    data = np.array(data)
    args = draw(titles()), bins, labels, data
    return args, Histogram(*args)


@composite
def empty_histograms(draw, dimension=0):
    if dimension <= 0:
        dimension = draw(sampled_from((1,2)))
    bins   = draw(bins_arrays(dimension=dimension))
    labels = draw(lists(text(characters, min_size=5), min_size=dimension, max_size=dimension))
    args   = draw(titles()), bins, labels
    return args, Histogram(*args)


@composite
def histograms_lists(draw, number=0, dimension=0, fixed_bins=None):
    if number <= 0:
        number = draw(integers(2, 5))
    empty_histogram  = empty_histograms (dimension=dimension)
    filled_histogram = filled_histograms(dimension=dimension, fixed_bins=fixed_bins)
    args, histograms = zip(*[ draw(one_of(empty_histogram, filled_histogram)) for i in range(number) ])

    titles, *_ = zip(*args)
    assume(len(set(titles)) == len(titles))

    return args, histograms

@mark.skip(reason="Delaying elimination of solid cities")
@given(bins_arrays())
@settings(deadline=None)
def test_histogram_initialization(bins):
    label      = [ 'Random distribution' ]
    title      = 'Test_histogram'
    test_histo = Histogram(title, bins, label)

    assert np.all     (a == b for a, b in zip(test_histo.bins, bins))
    assert np.allclose(test_histo.data     , np.zeros(shape=tuple(len(x) - 1 for x in bins)))
    assert np.allclose(test_histo.errors   , np.zeros(shape=tuple(len(x) - 1 for x in bins)))
    assert np.allclose(test_histo.out_range, np.zeros(shape=(2, len(bins)))                 )
    assert             test_histo.labels  == label
    assert             test_histo.title   == title

@mark.skip(reason="Delaying elimination of solid cities")
@given(bins_arrays())
@settings(deadline=None)
def test_histogram_initialization_with_values(bins):
    label     = [ 'Random distribution' ]
    title     = 'Test_histogram'
    data      = []
    out_range = []
    for ibin in bins:
        lower_limit = ibin[0]  * 0.95
        upper_limit = ibin[-1] * 1.05
        data.append(     np.random.uniform(lower_limit, upper_limit, 500)  )
        out_range.append(np.array([ np.count_nonzero(data[-1] < ibin[0]),
                                    np.count_nonzero(data[-1] > ibin[-1])]))

    data         = np.array(data)
    out_range    = np.array(out_range).T
    binned_data  = np.histogramdd(data.T, bins)[0]
    test_histo   = Histogram(title, bins, label, data)

    assert np.all     (a == b for a, b in zip(test_histo.bins, bins))
    assert np.allclose(test_histo.data     , binned_data         )
    assert np.allclose(test_histo.out_range, out_range           )
    assert np.allclose(test_histo.errors   , np.sqrt(binned_data))
    assert             test_histo.labels  == label
    assert             test_histo.title   == title

@mark.skip(reason="Delaying elimination of solid cities")
@given(empty_histograms())
@settings(deadline=None)
def test_histogram_fill(empty_histogram):
    _, test_histogram  = empty_histogram
    histobins          =  test_histogram.bins
    n_points           =   np.random.randint(5, 201)
    test_data          = [ np.random.uniform( bins[0]  * (1 + np.random.uniform()),
                                              bins[-1] * (1 + np.random.uniform()),
                                              n_points) for bins in histobins ]
    test_histogram.fill(test_data)
    out_of_range       = [ np.array([ np.count_nonzero(test_data[i] < bins[0] ),
                                      np.count_nonzero(test_data[i] > bins[-1]) ])
                                      for i, bins in enumerate(histobins) ]
    test_out_of_range  = np.array(out_of_range).T
    test_data          = np.histogramdd(np.asarray(test_data).T, test_histogram.bins)[0]
    test_errors        = np.sqrt(test_data)

    assert np.allclose(test_histogram.data     , test_data        )
    assert np.allclose(test_histogram.errors   , test_errors      )
    assert np.allclose(test_histogram.out_range, test_out_of_range)

@mark.skip(reason="Delaying elimination of solid cities")
@given(empty_histograms())
@settings(deadline=None)
def test_histogram_fill_with_weights(empty_histogram):
    _, test_histogram = empty_histogram
    histobins         =  test_histogram.bins
    n_points          =   np.random.randint(50, 201)
    test_data         = [ np.random.uniform( bins[0]  * (1 + np.random.uniform()),
                                             bins[-1] * (1 + np.random.uniform()),
                                             n_points) for bins in histobins ]
    test_weights      =   np.random.uniform(1, 10, n_points)
    test_histogram.fill(test_data, test_weights)
    out_of_range      = [ np.array([ np.count_nonzero(test_data[i] < bins[0] ),
                                     np.count_nonzero(test_data[i] > bins[-1]) ])
                                     for i, bins in enumerate(histobins) ]
    test_out_of_range = np.array(out_of_range).T
    test_data         = np.histogramdd(np.asarray(test_data).T, test_histogram.bins, weights=test_weights)[0]
    test_errors       = np.sqrt(test_data)

    assert np.allclose(test_histogram.data     , test_data        )
    assert np.allclose(test_histogram.errors   , test_errors      )
    assert np.allclose(test_histogram.out_range, test_out_of_range)

@mark.skip(reason="Delaying elimination of solid cities")
def test_bin_data():
    bins           = [ np.linspace(0., 5., 6) ]
    label          = [ 'Random distribution'  ]
    title          =   'Test_histogram'
    data           = np.array([-1., 2.2, 3.2, 4.5, 6.3, 7.1, 4.9, 3.1, 0.2, 2.1, 2.2, 1.1])
    test_data      = [1, 1, 3, 2, 2]
    test_out_range = [[1], [2]]

    test_histo             = Histogram(title, bins, label)
    binned_data, out_range = test_histo.bin_data(data, data_weights=np.ones(len(data)))

    assert np.allclose(binned_data, test_data     )
    assert np.allclose(out_range  , test_out_range)

@mark.skip(reason="Delaying elimination of solid cities")
def test_count_out_of_range():
    bins           = [ np.linspace(0., 5., 6) ]
    label          = [ 'Random distribution'  ]
    title          =    'Test_histogram'
    data           = np.array([-1., 2.2, 3.2, 4.5, 6.3, 7.1, 4.9, 3.1, 0.2, 2.1, 2.2, 1.1])
    test_data      = [1, 1, 3, 2, 2]
    test_out_range = [[1], [2]]

    test_histo = Histogram(title, bins, label)
    out_range = test_histo.count_out_of_range(np.array(data, ndmin=2))

    assert np.allclose(out_range, test_out_range)

@mark.skip(reason="Delaying elimination of solid cities")
@given(filled_histograms())
@settings(deadline=None)
def test_update_errors(filled_histogram):
    _, test_histogram     = filled_histogram
    test_histogram.errors = None
    test_histogram.update_errors()
    assert np.allclose(test_histogram.errors, np.sqrt(test_histogram.data))

@mark.skip(reason="Delaying elimination of solid cities")
@given(filled_histograms())
@settings(deadline=None)
def test_update_errors_with_values(filled_histogram):
    _, test_histogram = filled_histogram
    new_errors        = np.random.uniform(0., 1000, size=test_histogram.data.shape)
    test_histogram.update_errors(new_errors)
    assert np.allclose(test_histogram.errors, new_errors)

@mark.skip(reason="Delaying elimination of solid cities")
@given(filled_histograms(fixed_bins=[[50, 900, 20]]),
       filled_histograms(fixed_bins=[[50, 900, 20]]))
@settings(deadline=None)
def test_add_histograms(first_histogram, second_histogram):
    _, test_histogram1 =  first_histogram
    _, test_histogram2 = second_histogram
    sum_histogram      =   test_histogram1 + test_histogram2

    assert np.all     (a == b for a, b in zip(sum_histogram.bins, test_histogram1.bins))
    assert np.all     (a == b for a, b in zip(sum_histogram.bins, test_histogram2.bins))
    assert np.allclose(sum_histogram.data     ,         test_histogram1.data        + test_histogram2.data        )
    assert np.allclose(sum_histogram.errors   , np.sqrt(test_histogram1.errors ** 2 + test_histogram2.errors ** 2))
    assert np.allclose(sum_histogram.out_range,         test_histogram1.out_range   + test_histogram2.out_range   )
    assert             sum_histogram.labels  ==         test_histogram1.labels
    assert             sum_histogram.title   ==         test_histogram1.title

@mark.skip(reason="Delaying elimination of solid cities")
@given(filled_histograms(fixed_bins=[[50, 900, 20]]),
       filled_histograms(fixed_bins=[[50, 900,  5]]))
@settings(deadline=None)
def test_add_histograms_with_incompatible_binning_raises_ValueError(first_histogram, second_histogram):
    _, test_histogram1 =  first_histogram
    _, test_histogram2 = second_histogram

    with raises(ValueError):
        sum_histogram = test_histogram1 + test_histogram2

@mark.skip(reason="Delaying elimination of solid cities")
@given(filled_histograms(fixed_bins=[[50, 900, 20], [20, 180, 15]]),
       filled_histograms(fixed_bins=[[50, 900, 20], [20, 180, 15]]))
@settings(deadline=None)
def test_add_histograms_2d(first_histogram, second_histogram):
    _, test_histogram1 =  first_histogram
    _, test_histogram2 = second_histogram
    sum_histogram      =   test_histogram1 + test_histogram2

    assert np.all     (a == b for a, b in zip(sum_histogram.bins, test_histogram1.bins))
    assert np.all     (a == b for a, b in zip(sum_histogram.bins, test_histogram2.bins))
    assert np.allclose(sum_histogram.data     ,         test_histogram1.data        + test_histogram2.data        )
    assert np.allclose(sum_histogram.errors   , np.sqrt(test_histogram1.errors ** 2 + test_histogram2.errors ** 2))
    assert np.allclose(sum_histogram.out_range,         test_histogram1.out_range   + test_histogram2.out_range   )
    assert             sum_histogram.labels  ==         test_histogram1.labels
    assert             sum_histogram.title   ==         test_histogram1.title

@mark.skip(reason="Delaying elimination of solid cities")
@given   (histograms_lists())
@settings(deadline=None)
def test_histomanager_initialization_with_histograms(histogram_list):
    _, list_of_histograms = histogram_list
    histogram_manager     = HistoManager(list_of_histograms)

    for histogram in list_of_histograms:
        assert_histogram_equality(histogram, histogram_manager[histogram.title])

@mark.skip(reason="Delaying elimination of solid cities")
def test_histomanager_initialization_without_histograms():
    histogram_manager = HistoManager()
    assert len(histogram_manager.histos) == 0

@mark.skip(reason="Delaying elimination of solid cities")
@given(one_of(empty_histograms(), filled_histograms()))
@settings(deadline=None)
def test_new_histogram_in_histomanager(test_histogram):
    _, histogram      = test_histogram
    histoname         = histogram.title
    histogram_manager = HistoManager()
    histogram_manager.new_histogram(histogram)

    assert_histogram_equality(histogram, histogram_manager[histoname])

@mark.skip(reason="Delaying elimination of solid cities")
@given   (histograms_lists())
@settings(deadline=None)
def test_fill_histograms_in_histomanager(histogram_list):
    args, list_of_histograms = histogram_list
    titles, histobins, *_    = zip(*args)
    histogram_manager        = HistoManager(list_of_histograms)

    test_data_values  = {}
    old_data_values   = {}
    test_out_of_range = {}
    old_out_of_range  = {}
    for i, title in enumerate(titles):
        old_data_values  [title] =   np.copy(histogram_manager[title].data     )
        old_out_of_range [title] =   np.copy(histogram_manager[title].out_range)
        n_points                 =   np.random.randint(5, 201)
        test_data                = [ np.random.uniform( bins[0]  * (1 + np.random.uniform()),
                                                        bins[-1] * (1 + np.random.uniform()),
                                                        n_points)  for bins in histobins[i] ]
        test_data_values [title] = test_data
        out_of_range             = [ np.array([ np.count_nonzero(test_data[j] < bins[0] ),
                                                np.count_nonzero(test_data[j] > bins[-1]) ])
                                                for j, bins in enumerate(histobins[i])     ]
        test_out_of_range[title] =   np.array(out_of_range).T

    histogram_manager.fill_histograms(test_data_values)

    for histoname, data in test_data_values.items():
        histogram = histogram_manager       [histoname]
        old_data  = np.array(old_data_values[histoname])
        test_data = np.histogramdd(np.asarray(data).T, histogram.bins)[0]
        assert np.allclose(histogram.data     ,         test_data                    + old_data                   )
        assert np.allclose(histogram.errors   , np.sqrt(test_data                    + old_data)                  )
        assert np.allclose(histogram.out_range,         test_out_of_range[histoname] + old_out_of_range[histoname])

@mark.skip(reason="Delaying elimination of solid cities")
@given   (histograms_lists())
@settings(deadline=None)
def test_getitem_histomanager(histogram_list):
    args, list_of_histograms = histogram_list
    titles, histobins, *_    = zip(*args)
    histogram_manager        = HistoManager(list_of_histograms)

    for histoname in histogram_manager.histos:
        assert np.all     (a == b for a, b in zip(histogram_manager.histos[histoname].bins, histogram_manager[histoname].bins))
        assert np.allclose(histogram_manager.histos[histoname].data     , histogram_manager[histoname].data     )
        assert np.allclose(histogram_manager.histos[histoname].errors   , histogram_manager[histoname].errors   )
        assert np.allclose(histogram_manager.histos[histoname].out_range, histogram_manager[histoname].out_range)
        assert             histogram_manager.histos[histoname].labels  == histogram_manager[histoname].labels
        assert             histogram_manager.histos[histoname].title   == histogram_manager[histoname].title

@mark.skip(reason="Delaying elimination of solid cities")
@given(histograms_lists())
@settings(deadline=None)
def test_setitem_histomanager(histogram_list):
    args, list_of_histograms = histogram_list
    titles, histobins, *_    = zip(*args)
    histogram_manager1       = HistoManager()
    histogram_manager2       = HistoManager()

    for histogram in list_of_histograms:
        histoname                            = histogram.title
        histogram_manager1.histos[histoname] = histogram
        histogram_manager2       [histoname] = histogram

        assert_histogram_equality(histogram_manager1[histoname], histogram_manager2[histoname])
