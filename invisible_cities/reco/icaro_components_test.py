import numpy         as np
import numpy.testing as npt
import pandas        as pd

from hypothesis.strategies import integers
from hypothesis.strategies import floats
from hypothesis            import given
from pytest                import mark

from .. core.testing_utils import assert_dataframes_equal
from .. core               import core_functions   as core

from .                     import icaro_components as icarcomp


@mark.parametrize("signal", (icarcomp.type_of_signal.nS1, icarcomp.type_of_signal.nS2))
@given(nsignals= integers(min_value = 1,
                          max_value = 1e4))
def test_selection_nS_mask_and_checking_right_output(nsignals, signal):
    nevt = int(1e4)
    data = np.concatenate([np.zeros(nevt- nsignals),
                          np.ones  (nsignals)])
    np.random.shuffle(data)

    data = pd.DataFrame({'nS1': data, 'nS2': data,'event': range(nevt)})

    mask = icarcomp.selection_nS_mask_and_checking(data, signal)

    assert np.sum(mask) == nsignals


@given(integers(min_value = 1,
                max_value = 1e4))
def test_selection_nS_mask_and_checking_consistency(nsignals):
    nevt = int(1e4)
    data = np.concatenate([np.zeros(nevt- nsignals),
                          np.ones  (nsignals)])
    np.random.shuffle(data)

    data = pd.DataFrame({'nS1': data, 'event': range(nevt)})

    mask    = icarcomp.selection_nS_mask_and_checking(data, icarcomp.type_of_signal.nS1)
    mask_re = icarcomp.selection_nS_mask_and_checking(data, icarcomp.type_of_signal.nS1, input_mask=mask)
    npt.assert_equal(mask, mask_re)


@given(integers(min_value = 1,
                max_value = 1e4),
       integers(min_value = 1,
                max_value = 1e4))
def test_selection_nS_mask_and_checking_concatenating(ns1, ns2):
    nevt = int(1e4)
    dataS1 = np.concatenate([np.zeros(nevt- ns1),
                             np.ones (ns1)])
    dataS2 = np.concatenate([np.zeros(nevt- ns2),
                             np.ones (ns2)])
    np.random.shuffle(dataS1)
    np.random.shuffle(dataS2)

    data   = pd.DataFrame({'nS1': dataS1, 'nS2': dataS2, 'event': range(nevt)})
    maskS1 = icarcomp.selection_nS_mask_and_checking(data, icarcomp.type_of_signal.nS1)
    maskS2 = icarcomp.selection_nS_mask_and_checking(data, icarcomp.type_of_signal.nS2, maskS1)

    assert_dataframes_equal(data[maskS2], data[maskS1][data[maskS1].nS2 ==1])


@given(integers(min_value = 1,
                max_value = 1e4),
       floats  (min_value = 0,
                max_value = 0.9),
       floats  (min_value = 0.1,
                max_value = 1) )
def test_selection_nS_mask_and_checking_range_assertion(ns1, lim1, lim2):
    nevt = int(1e4)
    dataS1 = np.concatenate([np.zeros(nevt- ns1),
                             np.ones (ns1)])
    np.random.shuffle(dataS1)

    data   = pd.DataFrame({'nS1': dataS1, 'event': range(nevt)})

    eff = ns1 / 1e4
    min_eff = min(lim1, lim2)
    max_eff = max(lim1, lim2)
    if (eff < min_eff) or (eff > max_eff):
        npt.assert_raises(core.ValueOutOfRange,
                          icarcomp.selection_nS_mask_and_checking,
                          data,  icarcomp.type_of_signal.nS1, None,
                          [min_eff, max_eff], icarcomp.Strictness.stop_proccess)
