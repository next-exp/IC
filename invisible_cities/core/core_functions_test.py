import re
from time      import sleep
from functools import partial

import pandas as pd
import numpy  as np
import numpy.testing as npt

from pytest import approx
from pytest import mark
from pytest import raises

from flaky                  import flaky
from hypothesis             import given
from hypothesis.strategies  import integers
from hypothesis.strategies  import floats
from hypothesis.strategies  import sampled_from
from hypothesis.strategies  import composite
from hypothesis.extra.numpy import arrays

from .testing_utils import random_length_float_arrays
from .              import core_functions   as core
from .              import core_functions_c as core_c
from .              import  fit_functions   as fitf


sane_floats          = partial(floats, allow_nan=False, allow_infinity=False)
sorted_unique_arrays = random_length_float_arrays(min_length = 1, max_length = 10)
sorted_unique_arrays = sorted_unique_arrays.map(np.sort)
sorted_unique_arrays = sorted_unique_arrays.filter(lambda x: np.all(np.diff(x) > 0))


def test_timefunc(capfd):
    # We run a function with a defined time duration (sleep) and we check
    # the decorator prints a correct measurement.
    time = 0.12
    core.timefunc(sleep)(time)

    out, err = capfd.readouterr()
    time_measured = re.search(r'\d+\.\d+', out).group(0)
    time_measured = float(time_measured)
    np.isclose(time, time_measured)


def test_flat():
    inner_len = 12
    outer_len =  5
    nested_list = [[n for n in range(inner_len)] for i in range(outer_len)]
    flattened = core.flat(nested_list)
    for i in range(outer_len):
        for j in range(inner_len):
            assert flattened[j + i * inner_len] == nested_list[i][j]


def test_lrange():
    assert core.lrange(10) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_trange():
    assert core.trange(10) == (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)


@given(random_length_float_arrays())
def test_in_range_infinite(data):
    assert core.in_range(data).all()


@given(random_length_float_arrays(mask = lambda x: ((x<-10) or
                                               (x>+10) )))
def test_in_range_with_hole(data):
    assert not core.in_range(data, -10, 10).any()


def test_in_range_positives():
    data = np.linspace(-10., 10., 1001)
    assert np.count_nonzero(core.in_range(data, 0, 10)) == 500


@given(random_length_float_arrays(max_length = 100))
def test_in_range_right_shape(data):
    assert core.in_range(data, -1., 1.).shape == data.shape


@given(sorted_unique_arrays)
def test_in_range_left_open_interval(data):
    # right doesn't matter because it's infinite
    output = core.in_range(data, minval=data[0], maxval=np.inf, left_closed=False)
    assert not output[0]
    assert all(output[1:])


@given(sorted_unique_arrays)
def test_in_range_right_open_interval(data):
    # left doesn't matter because it's infinite
    output = core.in_range(data, minval=-np.inf, maxval=data[-1], right_closed=False)
    assert not output[-1]
    assert all(output[:-1])


@given(sorted_unique_arrays)
def test_in_range_left_close_interval(data):
    # right doesn't matter because it's infinite
    output = core.in_range(data, minval=data[0], maxval=np.inf, left_closed=True)
    assert all(output)


@given(sorted_unique_arrays)
def test_in_range_right_close_interval(data):
    # left doesn't matter because it's infinite
    output = core.in_range(data, minval=-np.inf, maxval=data[-1], right_closed=True)
    assert all(output)


@mark.parametrize(" first  second       norm_mode        expected".split(),
                  ((  1  ,    2  , core.NormMode.first ,   -1    ),
                   (  4  ,    2  , core.NormMode.second,    1    ),
                   (  4  ,    1  , core.NormMode.sumof ,    0.6  ),
                   (  8  ,    2  , core.NormMode.mean  ,    1.2  )))
def test_relative_difference(first, second, norm_mode, expected):
    got = core.relative_difference(first, second, norm_mode=norm_mode)
    assert got == approx(expected)


@mark.parametrize("norm_mode",
                  (0, "0", 1, "1", None, "None",
                   "first", "second",
                   "NormMode.first", "NormMode.second"))
def test_relative_difference_raises_TypeError(norm_mode):
    with raises(TypeError):
        core.relative_difference(1, 1, norm_mode=norm_mode)


@given(random_length_float_arrays(min_length = 5,
                                  min_value  = 1e-4,
                                  max_value  = 1e+4),
       floats(min_value = 1e-5,
              max_value = 1e-0))
def test_weighted_mean_and_var_all_weights_equal(data, weights):
    weights = np.full_like(data, weights)

    expected_mean, expected_var = np.mean(data), np.var(data)
    actual_mean  , actual_var   = core.weighted_mean_and_var(data, weights)

    npt.assert_allclose(expected_mean, actual_mean, rtol=1e-5)
    npt.assert_allclose(expected_var , actual_var , rtol=1e-5, atol=1e-4)


@given(floats  (min_value = -100,
                max_value = +100),
       floats  (min_value =    1,
                max_value = +100),
       integers(min_value =  100,
                max_value = 1000))
def test_weighted_mean_and_var_gaussian_function(mu, sigma, ndata):
    data = np.linspace(mu - 5 * sigma,
                       mu + 5 * sigma,
                       ndata)
    weights = fitf.gauss(data, 1, mu, sigma)

    ave, var = core.weighted_mean_and_var(data, weights)
    npt.assert_allclose(mu      , ave, atol=1e-8)
    npt.assert_allclose(sigma**2, var, rtol=1e-4)


@flaky(max_runs   = 4,
       min_passes = 3)
def test_weighted_mean_and_var_unbiased_frequentist():
    mu, sigma, ndata = 100, 1, np.random.randint(5, 20)
    data = np.random.normal(mu, sigma, size=ndata)
    values, freqs = np.unique(data, return_counts=True)

    ave, var = core.weighted_mean_and_var(values, freqs, unbiased=True, frequentist=True)
    assert abs(mu - ave) * (ndata / var)**0.5 < 3


@flaky(max_runs   = 4,
       min_passes = 3)
@given(integers(min_value=5, max_value=20))
def test_weighted_mean_and_var_unbiased_reliability_weights(ndata):
    mu, sigma = 100, 1
    values = np.linspace(mu - 5 * sigma,
                         mu + 5 * sigma,
                         ndata)
    weights = fitf.gauss(values, 1, mu, sigma)

    ave, var = core.weighted_mean_and_var(values, weights, unbiased=True, frequentist=False)
    assert abs(mu - ave) * (ndata / var)**0.5 < 3


def test_loc_elem_1d():
    assert core.loc_elem_1d(np.array(core.lrange(10)), 5) == 5

@composite
def nonzero_floats(draw, min_value=None, max_value=None):
    sign      = draw(sampled_from((-1, +1)))
    magnitude = draw(floats(min_value=min_value, max_value=max_value))
    return sign * magnitude

# Need to make sure that the step size is not to small compared to the
# range, otherwise the array that needs to be generated may contain so
# many values that we run out of memory. Do this by bounding the
# range's absolute size from above and the step's absolute size from
# below.
np_range_strategies = dict(
    start =         floats(min_value=-100,   max_value=+100),
    stop  =         floats(min_value=-100,   max_value=+100),
    step  = nonzero_floats(min_value=   0.1, max_value=  10))

# Check that the difference between adjacent elements is constant, and
# compare to the behaviour of the real np.arange.
@given(**np_range_strategies)
def test_np_range(start, stop, step):
    x = core.np_range(start, stop, step)
    y = x[1:  ]
    z = x[ :-1]
    steps = y - z
    npt.assert_almost_equal(steps, step)
    npt.assert_array_equal(x, np.arange(start, stop, step))

# Check that the sum of the forward and reverse ranges is the same
# everywhere.
@given(**np_range_strategies)
def test_np_reverse_range(start, stop, step):
    forward = core.        np_range(start, stop, step)
    reverse = core.np_reverse_range(start, stop, step)
    summed = forward + reverse
    if len(summed):
        npt.assert_almost_equal(summed, summed[0])

@given(integers(min_value=0, max_value=99), sane_floats())
def test_np_constant(N, k):
    array = core.np_constant(N,k)
    assert len(array) == N
    assert all(array == k)


def test_to_row_vector():
    x = np.arange(5)
    assert core.to_row_vector(x).shape == (1, 5)


def test_to_col_vector():
    x = np.arange(5)
    assert core.to_col_vector(x).shape == (5, 1)


def test_dict_map():
    assert (core.dict_map(lambda x: x**2, {'a': 1, 'b': 2, 'c': 3, 'd':  4})
            ==                       {'a': 1, 'b': 4, 'c': 9, 'd': 16})

def test_df_map():
    d = {'q' : [-1, +1, -1],
         'mass' : [0.511, 105., 1776.],
         'spin' :[0.5, 0.5, 0.5]}

    leptons = pd.DataFrame(d,index=['e-', 'mu+', 'tau-'])
    l2 = core.df_map(lambda x: x*1000, leptons, 'mass')
    assert l2.mass.values[0] == 511

def test_dict_filter_by_value():
    core.dict_filter_by_value(lambda x: x>5,
      {'a':1,'b':20,'c':3,'d':40}) == {'b': 20, 'd': 40}

def test_dict_filter_by_key():
    core.dict_filter_by_key(lambda x: x in 'ac',
      {'a':1,'b':20,'c':3,'d':40}) == {'a': 1, 'c': 3}

def test_farray_from_string():
    core.farray_from_string('1 10 100')[2] == 100

def test_rebin_array():
    """
    rebin arrays of len 1 to 100 with strides 1 to 10 and test output discarding the remainder
    """
    length = 100
    arr = np.ones(length)
    for stride in range(1,11):
        for s in range(length):
            for i, v in enumerate(core_c.rebin_array(arr[s:], stride, remainder=False)):
                if i == 0: assert v == min(stride, len(arr[s:]))
                else     : assert v == stride

def test_rebin_array_remainder():
    """
    rebin arrays of len 1 to 100 with strides 1 to 10 and test output with method=np.sum
    without discarding the remainder
    """
    length = 100
    arr = np.ones(length)
    for stride in range(1,11):
        for s in range(length):
            for i, v in enumerate(core_c.rebin_array(arr[s:], stride, remainder=True)):
                if i == 0:
                    assert v == min(stride, len(arr[s:]))
                elif i < len(arr[s:]) // stride:
                    assert v == stride
                else:
                    assert i == len(arr[s:]) // stride
                    assert v == min(stride, len(arr[s:][i * stride:]))

def test_define_window():
    mu, sigma = 100, 0.2 # mean and standard deviation
    sgn = np.random.normal(mu, sigma, 10000)
    n, _ = np.histogram(sgn, 50)
    n0, n1 = core.define_window(n, window_size=10)
    peak = core.loc_elem_1d(n, np.max(n))
    assert n0 == peak - 10
    assert n1 == peak + 10


@given(random_length_float_arrays(min_length = 1))
def test_mean_handle_empty_nonempty_input  (array):
    npt.assert_equal(  np.mean             (array),
                     core.mean_handle_empty(array))


def test_mean_handle_empty_empty_input():
    assert np.isnan(core.mean_handle_empty([]))


@given(random_length_float_arrays(min_length =  1,
                                  min_value  = -1e10,
                                  max_value  = +1e10))
def test_std_handle_empty_nonempty_input  (array):
    npt.assert_equal(  np.std             (array),
                     core.std_handle_empty(array))


def test_std_handle_empty_empty_input():
    assert np.isnan(core.std_handle_empty([]))


@given(arrays(float, 10, elements=floats(min_value=-1e5, max_value=1e5)))
def test_shift_to_bin_centers(x):
    x_shifted = core.shift_to_bin_centers(x)
    truth     = [np.mean(x[i:i+2]) for i in range(x.size-1)]
    npt.assert_allclose(x_shifted, truth, rtol=1e-6, atol=1e-6)


def test_binedges_from_bincenters():

    centers = np.array([1, 2, 3, 4])

    #no range
    binedges = core.binedges_from_bincenters(centers)
    expected = np.array([1, 1.5, 2.5, 3.5, 4])
    np.testing.assert_allclose(binedges, expected)

    #range
    binedges = core.binedges_from_bincenters(centers, range=(-10, 10))
    expected = np.array([-10, 1.5, 2.5, 3.5, 10])
    np.testing.assert_allclose(binedges, expected)


def test_binedges_from_bincenters_exceptions():

    # no range
    centers = np.array([2, 1, 2])
    with raises(ValueError, match="unsorted or repeated bin centers"):
        core.binedges_from_bincenters(centers)

    centers = np.array([1, 2, 3, 4, 3])
    with raises(ValueError, match="unsorted or repeated bin centers"):
        core.binedges_from_bincenters(centers)

    # range
    centers = np.array([1, 2, 3, 4])
    range = (2, 1)
    with raises(ValueError, match="lower edge must be lower than higher"):
        core.binedges_from_bincenters(centers, range=range)

    range = (1, 2)
    with raises(ValueError, match="bincenters out of range bounds"):
        core.binedges_from_bincenters(centers, range=range)


@given(x = arrays(np.float, 100, elements=floats(min_value=-500, max_value=500)),
       value = floats(min_value=-500, max_value=500))
def test_find_nearest(x, value):

    nearest = core.find_nearest(x, value)

    diff = np.abs(x-value)
    dmin = np.min(diff)

    idx = np.argwhere(dmin==diff)[0]
    assert x[idx] == nearest
