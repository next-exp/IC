"""
Tests for fit_functions
"""

import numpy as np
from pytest import mark
from numpy.testing import assert_array_equal, assert_allclose
from hypothesis import given
from hypothesis.strategies import integers, floats
from hypothesis.extra.numpy import arrays

import invisible_cities.core.fit_functions as fit


# TODO: Move to a different module {
def float_arrays(size       =   100,
                 min_value  = -1e20,
                 max_value  = +1e20,
                 mask       =  None,
                 **kwargs          ):
    elements = floats(min_value,
                      max_value,
                      **kwargs)
    if mask is not None:
        elements = elements.filter(mask)
    return arrays(dtype    = np.float32,
                  shape    =       size,
                  elements =   elements)


def FLOAT_ARRAY(*args, **kwargs):
    return float_arrays(*args, **kwargs).example()


def random_length_float_arrays(min_length =     0,
                               max_length =   100,
                               **kwargs          ):
    lengths = integers(min_length,
                       max_length)

    return lengths.flatmap(lambda n: float_arrays(       n,
                                                  **kwargs))
# }

@mark.slow
@given(random_length_float_arrays())
def test_in_range_infinite(data):
    assert fit.in_range(data).all()


@mark.slow
@given(random_length_float_arrays(mask = lambda x: ((x<-10) or
                                                    (x>+10) )))
def test_in_range_with_hole(data):
    assert not fit.in_range(data, -10, 10).any()


def test_in_range_positives():
    data = np.linspace(-10., 10., 1001)
    assert np.count_nonzero(fit.in_range(data, 0, 10)) == 500


@mark.slow
@given(random_length_float_arrays(max_length = 1000))
def test_in_range_right_shape(data):
    assert fit.in_range(data, -1., 1.).shape == data.shape


def test_get_errors():
    data = np.array([[ 9, -2,  1],
                     [-1,  4, -1],
                     [ 2, -4,  1]], dtype=float)
    assert_array_equal(fit.get_errors(data), [3., 2., 1.])


@given(float_arrays(min_value=-10,
                    max_value=10))
def test_gauss_symmetry(data):
    assert_allclose(fit.gauss(data, 1., 0., 1.),
                    fit.gauss(-data, 1., 0., 1.))


#def test_gauss_breaks_when_sigma_is_negative():
#    assert_raises(ValueError,
#                  fit.get_from_name,
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
    assert_allclose(fit.gauss(*args),
                    fit.gauss(*map(float, args)))


@given(random_length_float_arrays(min_length = 1,
                                  max_length = 10,))
def test_polynom_at_zero(pars):
    assert_allclose(fit.polynom(0., *pars),
                    pars[0])


@given(random_length_float_arrays(min_length = 1,
                                  max_length = 10,
                                  min_value  = 1e-8))
def test_polynom_at_one(pars):
    assert_allclose(fit.polynom(1., *pars),
                    np.sum(pars), rtol=1e-5)


@given(floats(min_value = -20.,
              max_value = -10.),
       floats(min_value = -20.,
              max_value = -10.))
def test_expo_double_negative_sign(x, mean):
    assert fit.expo(x, 1., mean) > fit.expo(x+1., 1., mean)


@given(floats(min_value = +10.,
              max_value = +20.),
       floats(min_value = +10.,
              max_value = +20.))
def test_expo_double_positive_sign(x, mean):
    assert fit.expo(x, 1., mean) < fit.expo(x+1., 1., mean)


@given(floats(min_value = -20.,
              max_value = -10.),
       floats(min_value = +10.,
              max_value = +20.))
def test_expo_flipped_signs(x, b):
    assert fit.expo(x, 1., b) < fit.expo(x+1., 1., b)


@given(integers(min_value = -10,
                max_value = +10),
       integers(min_value = -10,
                max_value = +10),
       integers(min_value = -10,
                max_value = +10).filter(lambda x: x != 0))
def test_expo_works_with_integers(x, a, b):
    args = [x, a, b]
    assert_allclose(fit.expo(*args),
                    fit.expo(*map(float, args)))


@given(floats(min_value = -10.,
              max_value = +10.),
       floats(min_value = -10.,
              max_value = +10.))
def test_power_at_one(a, b):
    assert_allclose(fit.power(1., a, b),
                    a)


@given(floats(min_value = -10.,
              max_value = +10.),
       floats(min_value =   0.,
              max_value = +10.).filter(lambda x: x != 0.))
def test_power_at_zero(a, b):
    assert_allclose(fit.power(0., a, b),
                    0.)


@mark.parametrize("        fn                 pars        ".split(),
                  ((fit.gauss         , (3.0,  2.0, 0.50)),
                   (fit.expo          , (6.0,  1.5)),
                   (fit.polynom       , (8.0, -0.5, 0.01)),
                   (fit.power         , (0.1,  0.8))))
def test_fit(fn, pars):
    pars = np.array(pars)
    x = np.arange(10.)
    y = fn(x, *pars)
    f = fit.fit(fn, x, y, pars * 1.5)
    assert_allclose(f.values, pars)


@mark.slow
@mark.parametrize(["func"],
                  ((fit.profileX,),
                   (fit.profileY,)))
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


@mark.parametrize("func xdata ydata".split(),
                  ((fit.profileX, FLOAT_ARRAY(100), FLOAT_ARRAY(100)),
                   (fit.profileY, FLOAT_ARRAY(100), FLOAT_ARRAY(100))))
def test_profile1D_custom_range(func, xdata, ydata):
    xrange     = (-100, 100)
    kw         = "yrange" if func.__name__.endswith("Y") else "xrange"
    d          = {kw: xrange}
    xp, yp, ye = func(xdata, ydata, **d)
    assert np.all(fit.in_range(xp, *xrange))


@mark.parametrize("func xdata ydata xrange".split(),
                  ((fit.profileX,
                    FLOAT_ARRAY(100, -100, 100),
                    FLOAT_ARRAY(100, -500,   0),
                    (-100, 100)),
                   (fit.profileY,
                    FLOAT_ARRAY(100, -100, 100),
                    FLOAT_ARRAY(100, -500,   0),
                    (-500, 0))))
def test_profile1D_full_range_x(func, xdata, ydata, xrange):
    xp, yp, ye = func(xdata, ydata)
    assert np.all(fit.in_range(xp, *xrange))

@mark.skip(reason="Hypothesis can't handle sequences longer than 2**10")
#@mark.parametrize("func xdata ydata".split(),
#                  ((fit.profileX,
#                    FLOAT_ARRAY(10000, -100, 100),
#                    FLOAT_ARRAY(10000, -500,   0)),
#                   (fit.profileY,
#                    FLOAT_ARRAY(10000, -100, 100),
#                    FLOAT_ARRAY(10000, -500,   0))))
def test_profile1D_one_bin_missing_x(func, xdata, ydata):
    xdata[fit.in_range(xdata, -2, 0)] += 5
    xp, yp, ye = func(xdata, ydata)
    assert xp.size == 99


@mark.parametrize("func xdata ydata".split(),
                  ((fit.profileX,
                    FLOAT_ARRAY(100),
                    FLOAT_ARRAY(100)),
                   (fit.profileY,
                    FLOAT_ARRAY(100),
                    FLOAT_ARRAY(100))))
def test_number_of_bins_matches(func, xdata, ydata):
    N = 50
    xp, yp, ye = func(xdata, ydata, N, drop_nan=False)
    assert xp.size == yp.size == ye.size == N


@mark.parametrize("func xdata ydata".split(),
                  ((fit.profileX,
                    FLOAT_ARRAY(100),
                    FLOAT_ARRAY(100)),
                   (fit.profileY,
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
    xp, yp, zp, ze = fit.profileXY(xdata,
                                   ydata,
                                   zdata,
                                   Nbin,
                                   Nbin)

    assert np.all(abs(zp - 0.5) < 3.00*rms)
    assert np.all(abs(ze - rms*(Nbin**2/N)**0.5) < eps)