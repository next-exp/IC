"""
Tests for fit_functions
"""

import numpy as np

from pytest        import mark
from pytest        import approx
from pytest        import raises
from flaky         import flaky
from numpy.testing import assert_array_equal

from .             import calib_functions as cf
from ..  core      import system_of_units as units


def test_bin_waveforms():
    bins = np.arange(0, 20)
    data = np.arange(1, 51).reshape(2, 25)
    comp = np.array([np.histogram(data[0], bins)[0],
                      np.histogram(data[1], bins)[0]])
    assert_array_equal(cf.bin_waveforms(data, bins), comp)


def test_spaced_integrals():
    limits = np.array([2, 4, 6])
    data = np.arange(20).reshape(2, 10)
    assert_array_equal(cf.spaced_integrals(data, limits),
                        np.array([[5, 9, 30], [25, 29, 70]]))


def test_int_limits_simple():
    samp = 1 * units.mus
    nint = 10
    start_int = 5
    wid_int = 1
    period = 50

    test_llim = np.array([  5,   6,  55,  56, 105, 106, 155, 156, 205, 206, 255, 256, 305, 306, 355, 356, 405, 406, 455, 456])
    test_dlim = np.array([  2,   3,  52,  53, 102, 103, 152, 153, 202, 203, 252, 253, 302, 303, 352, 353, 402, 403, 452, 453])

    llimits, dlimits = cf.int_limits(samp, nint, start_int, wid_int, period)

    assert_array_equal(llimits, test_llim)
    assert_array_equal(dlimits, test_dlim)


def test_filter_limits_inside():
    samp = 1 * units.mus
    nint = 10
    start_int = 5
    wid_int = 1
    period = 50

    fake_data_len = 500

    llimits, dlimits = cf.int_limits(samp, nint, start_int, wid_int, period)

    assert_array_equal(cf.filter_limits(llimits, fake_data_len), llimits)
    assert_array_equal(cf.filter_limits(dlimits, fake_data_len), dlimits)


def test_filter_limits_inside():
    samp = 1 * units.mus
    nint = 10
    start_int = 5
    wid_int = 1
    period = 50

    fake_data_len = 400

    llimits, dlimits = cf.int_limits(samp, nint, start_int, wid_int, period)

    fllimits = cf.filter_limits(llimits, fake_data_len)
    fdlimits = cf.filter_limits(dlimits, fake_data_len)

    assert len(fllimits) < len(llimits)
    assert len(fdlimits) < len(dlimits)
    assert len(fllimits) % 2 == 0
    assert len(fdlimits) % 2 == 0
    
    



