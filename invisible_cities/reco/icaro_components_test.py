import numpy         as np
import numpy.testing as npt
import pandas        as pd

from hypothesis.strategies import integers
from hypothesis.strategies import floats
from hypothesis            import given
from pytest                import mark
from pytest                import raises

from   sklearn.linear_model import RANSACRegressor

from .. core.testing_utils  import assert_dataframes_equal
from .. core                import core_functions   as core

from .                      import icaro_components as icarcomp


@mark.parametrize("signal", (icarcomp.type_of_signal.nS1, icarcomp.type_of_signal.nS2))
@given(nsignals= integers(min_value = 1,
                          max_value = 1e4))
def test_selection_nS_mask_and_checking_right_output(nsignals, signal):
    nevt = int(1e4)
    data = np.concatenate([np.zeros(nevt- nsignals), np.ones(nsignals)])
    np.random.shuffle(data)
    data = pd.DataFrame({'nS1': data, 'nS2': data, 'event': range(nevt)})

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

    assert np.sum(maskS1) >=  np.sum(maskS2)
    assert np.logical_not(maskS2[np.logical_not(maskS1)].all())


def test_selection_nS_mask_and_checking_range_assertion():
    nevt = int(1e4)
    ns1  = int(1e3)
    min_eff = 0.5
    max_eff = 1
    dataS1 = np.concatenate([np.zeros(nevt- ns1),
                             np.ones (ns1)])
    np.random.shuffle(dataS1)

    data   = pd.DataFrame({'nS1': dataS1, 'event': range(nevt)})

    eff = ns1 / 1e4

    with raises(core.ValueOutOfRange, match="values out of bounds"):
           icarcomp.selection_nS_mask_and_checking(data,  icarcomp.type_of_signal.nS1,
                                                   None,[min_eff, max_eff],
                                                   icarcomp.Strictness.stop_proccess)

@mark.parametrize("sigma", (0.1, 1, 5, 10, 20))
def test_sigma_estimation(sigma):
    nevt      = int(1e4)
    xrange    = [0, 1000]
    slope     = 100
    n0        = 100
    x         = np.random.uniform(*xrange, nevt)
    y         = slope * x + n0
    y         = np.random.normal(y, sigma)
    res_fit   = RANSACRegressor().fit(x.reshape(-1,1), y.reshape(-1, 1))
    sigma_est = icarcomp.sigma_estimation(x, y, res_fit)
    np.testing.assert_allclose(sigma, sigma_est, rtol=0.1)
