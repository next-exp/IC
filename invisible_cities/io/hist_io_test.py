import os
import string

from keyword import iskeyword

import numpy  as np
import tables as tb

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

from .. evm.histos          import HistoManager, Histogram
from .. io .hist_io         import save_histomanager_to_file
from .. io .hist_io         import get_histograms_from_file
from .. evm.histos_test     import histograms_lists
from .. evm.histos_test     import assert_histogram_equality


letters    = string.ascii_letters

@mark.skip(reason="Delaying elimination of solid cities")
@given   (histograms_lists(), text(letters, min_size=1))
@settings(deadline=None, max_examples=400)
def test_save_histomanager_to_file_write_mode(output_tmpdir, histogram_list, group):
    assume(not iskeyword(group))
    args, list_of_histograms = histogram_list
    histogram_manager        = HistoManager(list_of_histograms)

    file_out = os.path.join  (output_tmpdir    , 'test_save_histogram_manager.h5')
    save_histomanager_to_file(histogram_manager, file_out, group=group)

    with tb.open_file(file_out, "r") as h5in:
        file_group = getattr(h5in.root, group)
        for histogram in list_of_histograms:
            histoname    = histogram.title
            saved_labels = [str(label)[2:-1].replace('\\\\', '\\') for label in getattr(file_group, histoname + "_labels")[:]]

            assert histoname in file_group
            assert                                len(histogram.bins) == len(getattr(file_group, histoname + "_bins")[:])
            assert np.all     (a == b for a, b in zip(histogram.bins,        getattr(file_group, histoname + "_bins")[:]))
            assert np.allclose(histogram.data     , getattr(file_group, histoname              )[:])
            assert np.allclose(histogram.errors   , getattr(file_group, histoname + "_errors"  )[:])
            assert np.allclose(histogram.out_range, getattr(file_group, histoname + "_outRange")[:])
            assert             histogram.labels  == saved_labels


@mark.skip(reason="Delaying elimination of solid cities")
@given   (histograms_lists(), text(letters, min_size=1))
@settings(deadline=None, max_examples=400)
def test_save_histomanager_to_file_append_mode(output_tmpdir, histogram_list, group):
    assume(not iskeyword(group))
    args, list_of_histograms = histogram_list
    histogram_manager        = HistoManager(list_of_histograms[:1])

    file_out          = os.path.join(output_tmpdir, 'test_save_histogram_manager.h5')
    save_histomanager_to_file       (histogram_manager, file_out, mode='w', group=group)
    histogram_manager = HistoManager(list_of_histograms[1:])
    save_histomanager_to_file       (histogram_manager, file_out, mode='a', group=group)

    with tb.open_file(file_out, "r") as h5in:
        file_group = getattr(h5in.root, group)
        for histogram in list_of_histograms:
            histoname    = histogram.title
            saved_labels = [str(label)[2:-1].replace('\\\\', '\\') for label in getattr(file_group, histoname + "_labels")[:]]

            assert histoname in file_group
            assert                                len(histogram.bins) == len(getattr(file_group, histoname + "_bins")[:])
            assert np.all     (a == b for a, b in zip(histogram.bins,        getattr(file_group, histoname + "_bins")[:]))
            assert np.allclose(histogram.data     , getattr(file_group, histoname              )[:])
            assert np.allclose(histogram.errors   , getattr(file_group, histoname + "_errors"  )[:])
            assert np.allclose(histogram.out_range, getattr(file_group, histoname + "_outRange")[:])
            assert             histogram.labels  == saved_labels


@mark.skip(reason="Delaying elimination of solid cities")
@given   (histograms_lists(), text(letters, min_size=1), text(letters, min_size=1, max_size=1).filter(lambda x: x not in 'wa'))
@settings(deadline=None)
def test_save_histomanager_to_file_raises_ValueError(output_tmpdir, histogram_list, group, write_mode):
    args, list_of_histograms = histogram_list
    histogram_manager        = HistoManager(list_of_histograms)

    file_out   = os.path.join      (output_tmpdir, 'test_save_histogram_manager.h5')

    with raises(ValueError):
        save_histomanager_to_file(histogram_manager, file_out, mode=write_mode, group=group)


@mark.skip(reason="Delaying elimination of solid cities")
@given(histograms_lists())
@settings(deadline=None, max_examples=400)
def test_get_histograms_from_file(output_tmpdir, histogram_list):
    args, list_of_histograms  = histogram_list
    histogram_manager1        = HistoManager(list_of_histograms)

    file_out = os.path.join(output_tmpdir, 'test_save_histogram_manager.h5')
    save_histomanager_to_file(histogram_manager1, file_out)

    histogram_manager2 = get_histograms_from_file(file_out)

    assert len(histogram_manager1.histos) == len(histogram_manager2.histos)
    for histoname in histogram_manager1.histos:
        assert_histogram_equality(histogram_manager1[histoname], histogram_manager2[histoname])
