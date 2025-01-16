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

@mark.parametrize("nsignals", [0, 1, 100, 9999, 10*1000])
@mark.parametrize("signal"  , icarcomp.type_of_signal)
def test_select_nS_mask_and_check_right_output(nsignals, signal):
    nevt = 10*1000
    data = np.concatenate([np.zeros(nevt- nsignals), np.ones(nsignals)])
    np.random.shuffle(data)
    data = pd.DataFrame({'nS1': data, 'nS2': data, 'event': range(nevt)})
    mask = icarcomp.select_nS_mask_and_check(data, signal)
    assert np.count_nonzero(mask) == nsignals


@mark.parametrize("nsignals", [1, 100, 9999, 10*1000])
def test_select_nS_mask_and_check_consistency(nsignals):
    nevt = 10*1000
    data = np.concatenate([np.zeros(nevt - nsignals), np.ones(nsignals)])
    np.random.shuffle(data)
    data    = pd.DataFrame({'nS1': data, 'event': range(nevt)})
    mask    = icarcomp.select_nS_mask_and_check(data, icarcomp.type_of_signal.nS1)
    mask_re = icarcomp.select_nS_mask_and_check(data, icarcomp.type_of_signal.nS1, input_mask=mask)
    npt.assert_equal(mask, mask_re)


@mark.parametrize("ns1", [   1, 100, 9999, 10*1000])
@mark.parametrize("ns2", [0, 1, 100, 9999, 10*1000])
def test_select_nS_mask_and_check_concatenating(ns1, ns2):
    nevt   = 10*1000
    dataS1 = np.concatenate([np.zeros(nevt- ns1),
                             np.ones (ns1)])
    dataS2 = np.concatenate([np.zeros(nevt- ns2),
                             np.ones (ns2)])
    np.random.shuffle(dataS1)
    np.random.shuffle(dataS2)
    data   = pd.DataFrame({'nS1': dataS1, 'nS2': dataS2, 'event': range(nevt)})
    maskS1 = icarcomp.select_nS_mask_and_check(data, icarcomp.type_of_signal.nS1)
    maskS2 = icarcomp.select_nS_mask_and_check(data, icarcomp.type_of_signal.nS2, maskS1)

    assert np.count_nonzero(maskS1) >= np.count_nonzero(maskS2)
    assert np.logical_not(maskS2[np.logical_not(maskS1)].all())


def test_select_nS_mask_and_check_range_assertion():
    nevt    = 10*1000
    ns1     = 1000
    min_eff = 0.5
    max_eff = 1
    dataS1  = np.concatenate([np.zeros(nevt- ns1),
                              np.ones (ns1)])
    np.random.shuffle(dataS1)
    data   = pd.DataFrame({'nS1': dataS1, 'event': range(nevt)})
    eff    = ns1 / nevt

    with raises(core.ValueOutOfRange, match="values out of bounds"):
                icarcomp.select_nS_mask_and_check(data,  icarcomp.type_of_signal.nS1,
                                                  None,[min_eff, max_eff],
                                                  icarcomp.Strictness.raise_error)

@mark.parametrize("sigma", (0.5, 1, 5, 10, 20))
def test_estimate_sigma(sigma):
    nevt      = 10*1000
    xrange    = [0, 1000]
    slope     = 100
    n0        = 100
    x         = np.random.uniform(*xrange, nevt)
    y         = slope * x + n0
    y         = np.random.normal(y, sigma)
    res_fit   = RANSACRegressor().fit(x.reshape(-1,1), y.reshape(-1, 1))
    sigma_est = icarcomp.estimate_sigma(x, y, res_fit)
    np.testing.assert_allclose(sigma, sigma_est, rtol=0.1)
