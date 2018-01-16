"""
Tests for fit_functions
"""

import numpy as np

from pytest        import mark
from pytest        import approx
from pytest        import raises
from flaky         import flaky
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from hypothesis            import given
from hypothesis.strategies import integers
from hypothesis.strategies import floats
from . testing_utils       import float_arrays
from . testing_utils       import FLOAT_ARRAY
from . testing_utils       import random_length_float_arrays

from .                import core_functions as core
from .                import  fit_functions as fitf
from . stat_functions import poisson_sigma


def test_get_errors():
    data = np.array([[ 9, -2,  1],
                     [-1,  4, -1],
                     [ 2, -4,  1]], dtype=float)
    assert_array_equal(fitf.get_errors(data), [3., 2., 1.])


def test_get_chi2_and_pvalue_when_data_equals_error_and_fit_equals_zero():
    Nevt  = int(1e6)
    ydata = np.random.uniform(1, 100, Nevt)
    yfit  = np.zeros_like(ydata)
    errs  = ydata
    chi2, pvalue = fitf.get_chi2_and_pvalue(ydata, yfit, Nevt, errs)

    assert chi2   == approx(1  , rel=1e-3)
    assert pvalue == approx(0.5, rel=1e-3)


def test_get_chi2_and_pvalue_when_data_equals_fit():
    Nevt  = int(1e6)
    ydata = np.random.uniform(1, 100, Nevt)
    yfit  = ydata
    errs  = ydata**0.5 # Dummy value, not needed
    chi2, pvalue = fitf.get_chi2_and_pvalue(ydata, yfit, Nevt, errs)

    assert chi2   == approx(0., rel=1e-3)
    assert pvalue == approx(1., rel=1e-3)


@given(floats(min_value = -2500,
              max_value = +2500),
       floats(min_value = + 100,
              max_value = + 300))
def test_get_chi2_and_pvalue_gauss_errors(mean, sigma):
    Nevt  = int(1e6)
    ydata = np.random.normal(mean, sigma, Nevt)

    chi2, pvalue = fitf.get_chi2_and_pvalue(ydata, mean, Nevt-1, sigma)

    assert chi2 == approx(1, rel=1e-2)


@mark.parametrize("ey", (0, -1, -100))
def test_fit_raises_ValueError_when_negative_or_zero_value_in_sigma(ey):
    def dummy(x, m):
        return m

    x  = np.array([0  , 1  ])
    y  = np.array([4.1, 4.2])
    ey = np.full_like(y, ey)

    with raises(ValueError):
        fitf.fit(dummy, x, y, seed=(4,), sigma=ey)


def test_chi2_str_line():
    def line(x, m, n):
        return m * x + n

    y  = np.array([ 9.108e3, 10.34e3,   1.52387e5,   1.6202e5])
    ey = np.array([ 3.17   , 13.5   ,  70        ,  21       ])
    x  = np.array([29.7    , 33.8   , 481        , 511       ])

    f = fitf.fit(line, x, y, seed=(1,1), sigma=ey)

    assert f.chi2 == approx(14, rel=1e-02)


@mark.slow
@flaky(max_runs=10, min_passes=9)
def test_chi2_poisson_errors():
    mu    = np.random.uniform(-100, 100)
    sigma = np.random.uniform(   0, 100)
    A     =  1e9
    x     = np.linspace(mu - 5 * sigma, mu + 5 * sigma, 100)
    y     = fitf.gauss(x, A, mu, sigma)
    y     = np.random.poisson(y)
    errs  = poisson_sigma(y)

    f = fitf.fit(fitf.gauss, x, y, seed=(A, mu, sigma), sigma=errs)

    assert 0.60 < f.chi2 < 1.5


@mark.slow
@flaky(max_runs=10, min_passes=9)
def test_chi2_log_errors():
    mu    = np.random.uniform(-100, 100)
    sigma = np.random.uniform(   0, 100)
    A     =  1e9
    x     = np.linspace(mu - 2 * sigma, mu + 2*sigma, 100)
    y     = fitf.gauss(x, A, mu, sigma)
    y     = np.random.normal(y, np.log(y))
    errs  = np.log(y)

    f = fitf.fit(fitf.gauss, x, y, seed=(A, mu, sigma), sigma=errs)

    assert 0.60 < f.chi2 < 1.5


@given(float_arrays(min_value=-10,
                    max_value=10))
def test_gauss_symmetry(data):
    assert_allclose(fitf.gauss( data, 1., 0., 1.),
                    fitf.gauss(-data, 1., 0., 1.))


#def test_gauss_breaks_when_sigma_is_negative():
#    assert_raises(ValueError,
#                  fitf.get_from_name,
#                  funname)


@given(integers(min_value = -100,
                max_value = +100),
       integers(min_value = -100,
                max_value = +100),
       integers(min_value = -100,
                max_value = +100),
       integers(min_value = -1  ,
                max_value = +100))
def test_gauss_works_with_integers(x, a, b, c):
    args = [x, a , b, c]
    assert_allclose(fitf.gauss(*args),
                    fitf.gauss(*map(float, args)))


@given(random_length_float_arrays(min_length = 1,
                                  max_length = 10,))
def test_polynom_at_zero(pars):
    assert_allclose(fitf.polynom(0., *pars),
                    pars[0])


@given(random_length_float_arrays(min_length = 1,
                                  max_length = 10,
                                  min_value  = 1e-8))
def test_polynom_at_one(pars):
    assert_allclose(fitf.polynom(1., *pars),
                    np.sum(pars), rtol=1e-5)


@given(floats(min_value = -20.,
              max_value = -10.),
       floats(min_value = -20.,
              max_value = -10.))
def test_expo_double_negative_sign(x, mean):
    assert fitf.expo(x, 1., mean) > fitf.expo(x+1., 1., mean)


@given(floats(min_value = +10.,
              max_value = +20.),
       floats(min_value = +10.,
              max_value = +20.))
def test_expo_double_positive_sign(x, mean):
    assert fitf.expo(x, 1., mean) < fitf.expo(x+1., 1., mean)


@given(floats(min_value = -20.,
              max_value = -10.),
       floats(min_value = +10.,
              max_value = +20.))
def test_expo_flipped_signs(x, b):
    assert fitf.expo(x, 1., b) < fitf.expo(x+1., 1., b)


@given(integers(min_value = -10,
                max_value = +10),
       integers(min_value = -10,
                max_value = +10),
       integers(min_value = -10,
                max_value = +10).filter(lambda x: x != 0))
def test_expo_works_with_integers(x, a, b):
    args = [x, a, b]
    assert_allclose(fitf.expo(*args),
                    fitf.expo(*map(float, args)))


@given(floats(min_value = -10.,
              max_value = +10.),
       floats(min_value = -10.,
              max_value = +10.))
def test_power_at_one(a, b):
    assert_allclose(fitf.power(1., a, b),
                    a)


@given(floats(min_value = -10.,
              max_value = +10.),
       floats(min_value =   0.,
              max_value = +10.).filter(lambda x: x != 0.))
def test_power_at_zero(a, b):
    assert_allclose(fitf.power(0., a, b),
                    0.)


@mark.parametrize("        fn                 pars        ".split(),
                  ((fitf.gauss         , (3.0,  2.0, 0.50)),
                   (fitf.expo          , (6.0,  1.5)),
                   (fitf.polynom       , (8.0, -0.5, 0.01)),
                   (fitf.power         , (0.1,  0.8))))
def test_fit(fn, pars):
    pars = np.array(pars)
    x = np.arange(10.)
    y = fn(x, *pars)
    f = fitf.fit(fn, x, y, pars * 1.5)
    assert_allclose(f.values, pars)


def test_fit_reduced_range():
    pars = np.array([1, 20, 5])
    x = np.linspace(0, 50, 100)
    y = fitf.gauss(x, *pars)

    f1 = fitf.fit(fitf.gauss, x, y, pars * 1.5)
    f2 = fitf.fit(fitf.gauss, x, y, pars * 1.5, fit_range=(10, 30))
    assert_allclose(f1.values, f2.values)


@mark.parametrize("reduced".split(),
                  ((True ,),
                   (False,)))
def test_fit_with_errors(reduced):
    pars = np.array([1e3, 1e2, 1e1])
    x = np.linspace(100, 300, 100)
    y = fitf.gauss(x, *pars)
    e = 0.1 * y

    fit_range = (50, 150) if reduced else None
    f = fitf.fit(fitf.gauss, x, y, pars * 1.2, fit_range=fit_range, sigma=e, maxfev=10000)
    assert_allclose(f.values, pars)


@mark.parametrize(["func"],
                  ((fitf.profileX,),
                   (fitf.profileY,)))
def test_profile1D_uniform_distribution(func):
    N    = 100000
    Nbin = 100
    n    = N / Nbin
    rms  = 12**-0.5
    eps  = rms * n**0.5

    xdata = np.random.rand(N)
    ydata = np.random.rand(N)
    xp, yp, ye = func(xdata, ydata, Nbin)

    assert np.allclose(yp, 0.5, atol=3*rms)
    assert np.allclose(ye, rms/n**0.5, rtol=eps)


@mark.parametrize(["func"],
                  ((fitf.profileX,),
                   (fitf.profileY,)))
@flaky(max_runs=3, min_passes=1)
def test_profile1D_uniform_distribution_std(func):
    N    = 100000
    Nbin = 100
    rms  = 12**-0.5

    xdata = np.random.rand(N)
    ydata = np.random.rand(N)
    xp, yp, ye = func(xdata, ydata, Nbin, std=True)

    assert np.allclose(ye, rms, rtol=0.1)


@mark.parametrize("func xdata ydata".split(),
                  ((fitf.profileX, FLOAT_ARRAY(100), FLOAT_ARRAY(100)),
                   (fitf.profileY, FLOAT_ARRAY(100), FLOAT_ARRAY(100))))
def test_profile1D_custom_range(func, xdata, ydata):
    xrange     = (-100, 100)
    kw         = "yrange" if func.__name__.endswith("Y") else "xrange"
    d          = {kw: xrange}
    xp, yp, ye = func(xdata, ydata, **d)
    assert np.all(core.in_range(xp, *xrange))


@mark.parametrize("func xdata ydata xrange".split(),
                  ((fitf.profileX,
                    FLOAT_ARRAY(100, -100, 100),
                    FLOAT_ARRAY(100, -500,   0),
                    (-100, 100)),
                   (fitf.profileY,
                    FLOAT_ARRAY(100, -100, 100),
                    FLOAT_ARRAY(100, -500,   0),
                    (-500, 0))))
def test_profile1D_full_range_x(func, xdata, ydata, xrange):
    xp, yp, ye = func(xdata, ydata)
    assert np.all(core.in_range(xp, *xrange))

@mark.skip(reason="Hypothesis can't handle sequences longer than 2**10")
#@mark.parametrize("func xdata ydata".split(),
#                  ((fitf.profileX,
#                    FLOAT_ARRAY(10000, -100, 100),
#                    FLOAT_ARRAY(10000, -500,   0)),
#                   (fitf.profileY,
#                    FLOAT_ARRAY(10000, -100, 100),
#                    FLOAT_ARRAY(10000, -500,   0))))
def test_profile1D_one_bin_missing_x(func, xdata, ydata):
    xdata[core.in_range(xdata, -2, 0)] += 5
    xp, yp, ye = func(xdata, ydata)
    assert xp.size == 99


@mark.parametrize("func xdata ydata".split(),
                  ((fitf.profileX,
                    FLOAT_ARRAY(100, min_value=-1e10, max_value=+1e10),
                    FLOAT_ARRAY(100, min_value=-1e10, max_value=+1e10)),
                   (fitf.profileY,
                    FLOAT_ARRAY(100, min_value=-1e10, max_value=+1e10),
                    FLOAT_ARRAY(100, min_value=-1e10, max_value=+1e10))))
def test_number_of_bins_matches(func, xdata, ydata):
    N = 50
    xp, yp, ye = func(xdata, ydata, N, drop_nan=False)
    assert xp.size == yp.size == ye.size == N


@mark.parametrize("func xdata ydata".split(),
                  ((fitf.profileX,
                    FLOAT_ARRAY(100),
                    FLOAT_ARRAY(100)),
                   (fitf.profileY,
                    FLOAT_ARRAY(100),
                    FLOAT_ARRAY(100))))
def test_empty_dataset_yields_nans(func, xdata, ydata):
    N = 50
    empty = np.array([])
    xp, yp, ye = func(empty, empty, N,
                      (-100, 100),
                      (-100, 100),
                      drop_nan=False)
    assert np.all(np.isnan(yp))
    assert np.all(np.isnan(ye))


@mark.slow
def test_profileXY_full_range():
    N    = 10000
    Nbin = 100
    rms  = 12**-0.5
    eps  = 0.2*rms * (N/Nbin)**0.5

    xdata = np.random.rand(N)
    ydata = np.random.rand(N)
    zdata = np.random.rand(N)
    xp, yp, zp, ze = fitf.profileXY(xdata,
                                    ydata,
                                    zdata,
                                    Nbin,
                                    Nbin)

    assert np.all(abs(zp - 0.5) < 3.00*rms)
    assert np.all(abs(ze - rms*(Nbin**2/N)**0.5) < eps)
