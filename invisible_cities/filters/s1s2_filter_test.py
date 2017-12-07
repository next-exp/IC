import numpy as np

from pytest                 import fixture
from hypothesis             import given
from hypothesis.strategies  import lists
from hypothesis.strategies  import integers
from hypothesis.strategies  import booleans
from hypothesis.strategies  import composite

from .. evm.new_pmaps import  PMTResponses
from .. evm.new_pmaps import SiPMResponses
from .. evm.new_pmaps import S1
from .. evm.new_pmaps import S2
from .. evm.new_pmaps import PMap
from .  s1s2_filter   import S12SelectorOutput
from .  s1s2_filter   import S12Selector
from .  s1s2_filter   import pmap_filter


@composite
def random_filter_output(draw, n_s1, n_s2):
    ok = draw(booleans())
    s1 = draw(lists(booleans(), min_size=n_s1, max_size=n_s1))
    s2 = draw(lists(booleans(), min_size=n_s2, max_size=n_s2))
    return S12SelectorOutput(ok, s1, s2)


@composite
def anded_filter(draw):
    n_s1 = draw(integers(0, 5))
    n_s2 = draw(integers(0, 5))
    f_a  = draw(random_filter_output(n_s1, n_s2))
    f_b  = draw(random_filter_output(n_s1, n_s2))

    ok_anded = f_a.passed and f_b.passed
    s1_anded = [fa and fb for fa, fb in zip(f_a.s1_peaks, f_b.s1_peaks)]
    s2_anded = [fa and fb for fa, fb in zip(f_a.s2_peaks, f_b.s2_peaks)]
    f_anded  = S12SelectorOutput(ok_anded, s1_anded, s2_anded)
    return f_a, f_b, f_anded


@composite
def ored_filter(draw):
    n_s1 = draw(integers(0, 5))
    n_s2 = draw(integers(0, 5))
    f_a  = draw(random_filter_output(n_s1, n_s2))
    f_b  = draw(random_filter_output(n_s1, n_s2))

    ok_ored = f_a.passed or f_b.passed
    s1_ored = [fa or fb for fa, fb in zip(f_a.s1_peaks, f_b.s1_peaks)]
    s2_ored = [fa or fb for fa, fb in zip(f_a.s2_peaks, f_b.s2_peaks)]
    f_ored  = S12SelectorOutput(ok_ored, s1_ored, s2_ored)
    return f_a, f_b, f_ored


@fixture(scope="session")
def selector_conf():
    return dict(s1_nmin     =   1  , s1_nmax     = 1    ,
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


def build_peak(Pk, subwf, nsi=0):
    n      = subwf.size
    size   = 3 * n

    times  = np.arange(size, dtype=np.double)
    enes   = np.zeros (size, dtype=np.double)
    enes[n:2*n] = subwf

    pmt_r  =  PMTResponses([-1], [enes])
    sipm_r = SiPMResponses.build_empty_instance()
    ids    = np.random.randint(0, nsi + 1, size=nsi)
    sipm_r = SiPMResponses(ids, np.zeros((nsi, size)))
    return Pk(times, pmt_r, sipm_r)


@fixture(scope="session")
def true_s1_peak():
    """All variables ok."""
    enes = np.linspace(0, 0.5, 30)
    return build_peak(S1, enes)


@fixture(scope="session")
def small_s1_peak():
    """Total energy and width ok. Too small in height."""
    enes = np.linspace(0, 0.2, 30)
    return build_peak(S1, enes)


@fixture(scope="session")
def weak_s1_peak():
    """Height and width ok. Total energy below minimum."""
    enes     = np.zeros(30)
    enes[ 0] = enes[-1] = 0.11
    enes[15] =            0.44
    return build_peak(S1, enes)


@fixture(scope="session")
def short_s1_peak():
    """Total energy and height ok. Width too small."""
    enes = np.full(30, 1.5)
    return build_peak(S1, enes)


@fixture(scope="session")
def true_s2_peak():
    """All variables ok."""
    enes = np.linspace(0, 200, 300)
    return build_peak(S2, enes, 5)


@fixture(scope="session")
def small_s2_peak():
    """Total energy width ok. Too small in height."""
    enes = np.linspace(0, 50, 300)
    return build_peak(S2, enes, 5)


@fixture(scope="session")
def weak_s2_peak():
    """Height and width ok. Total energy below minimum."""
    enes      = np.zeros(300)
    enes[  0] = enes[-1] = 2.2
    enes[150] =            110
    return build_peak(S2, enes, 5)


@fixture(scope="session")
def short_s2_peak():
    """Total energy and height ok. Width too small."""
    enes = np.full(50, 500)
    return build_peak(S2, enes, 5)


@fixture(scope="session")
def nosipm_s2_peak():
    """Total energy and height ok. Width too small."""
    enes = np.linspace(0, 200, 300)
    return build_peak(S2, enes, 0)


@given(anded_filter())
def test_s12selectoroutput_and(filters):
    one, two, true = filters
    f_1 = S12SelectorOutput(one.passed, one.s1_peaks, one.s2_peaks)
    f_2 = S12SelectorOutput(two.passed, two.s1_peaks, two.s2_peaks)

    f_anded = f_1 & f_2
    assert       true.passed    ==      f_anded.passed
    assert tuple(true.s1_peaks) == tuple(f_anded.s1_peaks)
    assert tuple(true.s2_peaks) == tuple(f_anded.s2_peaks)


@given(ored_filter())
def test_s12selectoroutput_or(filters):
    one, two, true = filters
    f_1 = S12SelectorOutput(one.passed, one.s1_peaks, one.s2_peaks)
    f_2 = S12SelectorOutput(two.passed, two.s1_peaks, two.s2_peaks)

    f_ored = f_1 | f_2

    assert       true.passed    ==       f_ored.passed
    assert tuple(true.s1_peaks) == tuple(f_ored.s1_peaks)
    assert tuple(true.s2_peaks) == tuple(f_ored.s2_peaks)


def test_s12selector_select_s1(selector_conf,
                                true_s1_peak,
                               small_s1_peak,
                                weak_s1_peak,
                               short_s1_peak):
    selector = S12Selector(**selector_conf)
    peaks    = [true_s1_peak, small_s1_peak, weak_s1_peak, short_s1_peak]

    selector_output = selector.select_s1(peaks)
    truth           = (True, False, False, False)
    assert tuple(selector_output) == truth


def test_s12selector_select_s2(selector_conf,
                               true_s2_peak,
                               small_s2_peak,
                               weak_s2_peak,
                               short_s2_peak,
                               nosipm_s2_peak):
    selector = S12Selector(**selector_conf)
    peaks = [true_s2_peak, small_s2_peak, weak_s2_peak,
             short_s2_peak, nosipm_s2_peak]

    selector_output = selector.select_s2(peaks)
    truth           = (True, False, False, False, False)
    assert tuple(selector_output) == truth

def test_pmap_filter(selector_conf,
                      true_s1_peak,
                     small_s1_peak,
                      weak_s1_peak,
                     short_s1_peak,

                      true_s2_peak,
                     small_s2_peak,
                      weak_s2_peak,
                     short_s2_peak,
                    nosipm_s2_peak):
    selector = S12Selector(**selector_conf)
    s1_peaks = [true_s1_peak, small_s1_peak, weak_s1_peak, short_s1_peak]
    s2_peaks = [true_s2_peak, small_s2_peak, weak_s2_peak, short_s2_peak, nosipm_s2_peak]

    np.random.shuffle(s1_peaks)
    np.random.shuffle(s2_peaks)

    filter_output  = pmap_filter(selector, PMap(s1_peaks, s2_peaks))
    truth_s1_peaks = [pk is true_s1_peak for pk in s1_peaks]
    truth_s2_peaks = [pk is true_s2_peak for pk in s2_peaks]
    truth          = S12SelectorOutput(True, truth_s1_peaks, truth_s2_peaks)

    assert       filter_output.passed    ==       truth.passed
    assert tuple(filter_output.s1_peaks) == tuple(truth.s1_peaks)
    assert tuple(filter_output.s2_peaks) == tuple(truth.s2_peaks)
