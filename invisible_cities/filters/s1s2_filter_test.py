from argparse import Namespace

import numpy as np

from pytest                 import fixture
from hypothesis             import given
from hypothesis.strategies  import integers
from hypothesis.strategies  import booleans
from hypothesis.strategies  import dictionaries
from hypothesis.strategies  import composite

from .. evm.pmaps   import Peak
from .. evm.pmaps   import S1
from .. evm.pmaps   import S2
from .. evm.pmaps   import S2Si
from .  s1s2_filter import S12SelectorOutput
from .  s1s2_filter import S12Selector


@composite
def random_filter_output(draw):
    ok = draw(booleans())
    s1 = draw(dictionaries(keys     = integers(min_value=0, max_value=10),
                           values   = booleans(),
                           max_size = 5))
    s2 = draw(dictionaries(keys     = integers(min_value=0, max_value=10),
                           values   = booleans(),
                           max_size = 5))
    return S12SelectorOutput(ok, s1, s2)


@composite
def anded_filter(draw):
    filter_1 = draw(random_filter_output())
    filter_2 = draw(random_filter_output())

    ok_anded = filter_1.passed and filter_2.passed
    s1_anded = {}
    for peak_no in set(filter_1.s1_peaks) | set(filter_2.s1_peaks):
        s1_anded[peak_no] = (filter_1.s1_peaks.get(peak_no, False) and
                             filter_2.s1_peaks.get(peak_no, False))

    s2_anded = {}
    for peak_no in set(filter_1.s2_peaks) | set(filter_2.s2_peaks):
        s2_anded[peak_no] = (filter_1.s2_peaks.get(peak_no, False) and
                             filter_2.s2_peaks.get(peak_no, False))

    filter_anded = S12SelectorOutput(ok_anded, s1_anded, s2_anded)
    return filter_1, filter_2, filter_anded


@composite
def ored_filter(draw):
    filter_1 = draw(random_filter_output())
    filter_2 = draw(random_filter_output())

    ok_ored = filter_1.passed or filter_2.passed
    s1_ored = {}
    for peak_no in set(filter_1.s1_peaks) | set(filter_2.s1_peaks):
        s1_ored[peak_no] = (filter_1.s1_peaks.get(peak_no, False) or
                            filter_2.s1_peaks.get(peak_no, False))

    s2_ored = {}
    for peak_no in set(filter_1.s2_peaks) | set(filter_2.s2_peaks):
        s2_ored[peak_no] = (filter_1.s2_peaks.get(peak_no, False) or
                            filter_2.s2_peaks.get(peak_no, False))

    filter_ored = S12SelectorOutput(ok_ored, s1_ored, s2_ored)
    return filter_1, filter_2, filter_ored


@fixture(scope="session")
def selector_conf():
    return Namespace(s1_nmin     =   1  , s1_nmax     = 1    ,
                     s1_emin     =   2  , s1_emax     = 10   ,
                     s1_wmin     =  10  , s1_wmax     = 100  ,
                     s1_hmin     =   0.4, s1_hmax     = 10   ,
                     s1_ethr     =   0.1,
                     s2_nmin     =   1  , s2_nmax     = 2    ,
                     s2_emin     = 500  , s2_emax     = 50000,
                     s2_wmin     = 200  , s2_wmax     = 1000 ,
                     s2_hmin     = 100  , s2_hmax     = 1000 ,
                     s2_nsipmmin =   1  , s2_nsipmmax = 10   ,
                     s2_ethr     =   2  )


def empty_times_enes(size):
    return np.arange(size, dtype=np.double), np.zeros(size, dtype=np.double)


@fixture(scope="session")
def true_s1_peak(selector_conf):
    """All variables ok."""
    times, enes = empty_times_enes(100)
    enes[40:70] = np.linspace(0, 0.5, 30)
    peak        = Peak(times, enes)
    return peak


@fixture(scope="session")
def small_s1_peak(selector_conf):
    """Total energy and width ok. Too small in height."""
    times, enes = empty_times_enes(100)
    enes[40:70] = np.linspace(0, 0.2, 30)
    peak        = Peak(times, enes)
    return peak


@fixture(scope="session")
def weak_s1_peak(selector_conf):
    """Height and width ok. Total energy below minimum."""
    times, enes = empty_times_enes(100)
    enes[40]    = enes[70] = selector_conf.s1_ethr * 1.1
    enes[55]    = selector_conf.s1_hmin * 1.1
    peak        = Peak(times, enes)
    return peak


@fixture(scope="session")
def short_s1_peak(selector_conf):
    """Total energy and height ok. Width too small."""
    times, enes = empty_times_enes(100)
    enes[40:45] = 1.5
    peak        = Peak(times, enes)
    return peak


@fixture(scope="session")
def true_s2_peak(selector_conf):
    """All variables ok."""
    times, enes   = empty_times_enes(1000)
    enes[400:700] = np.linspace(0, 200, 300)
    peak          = Peak(times, enes)
    return peak


@fixture(scope="session")
def small_s2_peak(selector_conf):
    """Total energy width ok. Too small in height."""
    times, enes   = empty_times_enes(1000)
    enes[400:700] = np.linspace(0, 50, 300)
    peak          = Peak(times, enes)
    return peak


@fixture(scope="session")
def weak_s2_peak(selector_conf):
    """Height and width ok. Total energy below minimum."""
    times, enes = empty_times_enes(1000)
    enes[400]   = enes[700] = selector_conf.s2_ethr * 1.1
    enes[555]   = selector_conf.s2_hmin * 1.1
    peak        = Peak(times, enes)
    return peak


@fixture(scope="session")
def short_s2_peak(selector_conf):
    """Total energy and height ok. Width too small."""
    times, enes   = empty_times_enes(1000)
    enes[400:450] = 500#selector_conf.s2_emin
    peak          = Peak(times, enes)
    return peak


@fixture(scope="session")
def true_s2si_peak(selector_conf, nsi=5):
    """Number of sipms ok"""
    si = {}
    for i in range(nsi):
        si[i] = empty_times_enes(1000)[1]
        si[i][400:700] = np.random.rand(300)
    return si


@fixture(scope="session")
def fake_s2si_peak(selector_conf):
    """Empty peak"""
    return {}


@given(anded_filter())
def test_s12selectoroutput_and(filters):
    one, two, true = filters
    f_1 = S12SelectorOutput(one.passed, one.s1_peaks, one.s2_peaks)
    f_2 = S12SelectorOutput(two.passed, two.s1_peaks, two.s2_peaks)

    f_anded = f_1 & f_2
    assert true.passed   == f_anded.passed
    assert true.s1_peaks == f_anded.s1_peaks
    assert true.s2_peaks == f_anded.s2_peaks


@given(ored_filter())
def test_s12selectoroutput_or(filters):
    one, two, true = filters
    f_1 = S12SelectorOutput(one.passed, one.s1_peaks, one.s2_peaks)
    f_2 = S12SelectorOutput(two.passed, two.s1_peaks, two.s2_peaks)

    f_ored = f_1 | f_2

    assert true.passed   == f_ored.passed
    assert true.s1_peaks == f_ored.s1_peaks
    assert true.s2_peaks == f_ored.s2_peaks


def test_s12selector_select_s1(selector_conf,
                                true_s1_peak,
                               small_s1_peak,
                                weak_s1_peak,
                               short_s1_peak):
    selector = S12Selector(**selector_conf.__dict__)
    peaks = dict(enumerate([( true_s1_peak.t,  true_s1_peak.E),
                            (small_s1_peak.t, small_s1_peak.E),
                            ( weak_s1_peak.t,  weak_s1_peak.E),
                            (short_s1_peak.t, short_s1_peak.E)]))

    selector_output = selector.select_s1(S1(peaks))
    truth           = dict(enumerate([True, False, False, False]))
    assert selector_output == truth


def test_s12selector_select_s2_true_si_peak(selector_conf,
                                             true_s2_peak,
                                            small_s2_peak,
                                             weak_s2_peak,
                                            short_s2_peak,
                                             true_s2si_peak):
    selector = S12Selector(**selector_conf.__dict__)
    peaks = dict(enumerate([( true_s2_peak.t,  true_s2_peak.E),
                            (small_s2_peak.t, small_s2_peak.E),
                            ( weak_s2_peak.t,  weak_s2_peak.E),
                            (short_s2_peak.t, short_s2_peak.E)]))
    sipms = {p: true_s2si_peak for p in peaks}

    selector_output = selector.select_s2(S2(peaks), S2Si(peaks, sipms))
    truth           = dict(enumerate([True, False, False, False]))
    assert selector_output == truth


def test_s12selector_select_s2_fake_si_peak(selector_conf,
                                             true_s2_peak,
                                            small_s2_peak,
                                             weak_s2_peak,
                                            short_s2_peak,
                                             fake_s2si_peak):
    selector = S12Selector(**selector_conf.__dict__)
    peaks = dict(enumerate([( true_s2_peak.t,  true_s2_peak.E),
                            (small_s2_peak.t, small_s2_peak.E),
                            ( weak_s2_peak.t,  weak_s2_peak.E),
                            (short_s2_peak.t, short_s2_peak.E)]))
    sipms = {p: fake_s2si_peak for p in peaks}

    selector_output = selector.select_s2(S2(peaks), S2Si(peaks, sipms))
    truth           = dict(enumerate([False, False, False, False]))
    assert np.all(selector_output == truth)
