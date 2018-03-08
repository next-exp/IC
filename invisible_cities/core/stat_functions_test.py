import numpy as np

from pytest import mark
from pytest import approx

from hypothesis             import given
from hypothesis.strategies  import composite
from hypothesis.strategies  import integers
from hypothesis.strategies  import floats
from hypothesis.strategies  import sampled_from
from hypothesis.extra.numpy import arrays

from . stat_functions import poisson_factor
from . stat_functions import poisson_sigma


@composite
def arrays_of_positive_numbers(draw):
    _strategies     = {int: integers, float: floats}
    _type, strategy = draw(sampled_from(list(_strategies.items())))
    size            = draw(integers(1, 10))
    return draw(arrays(_type, size, strategy(0, 100)))


@mark.parametrize("mean", np.linspace(0.1, 3, 3))
def test_poisson_factor_mean(mean):
    k        = 1
    expected = np.exp(-mean) * mean
    actual   = poisson_factor(k, mean)

    assert actual == approx(expected)


@mark.parametrize("k", range(10))
def test_poisson_factor_k(k):
    mean  = 1
    first = poisson_factor(0, mean)
    nth   = poisson_factor(k, mean)
    assert first / nth == approx(np.math.factorial(k))


@mark.parametrize("default", (0, 1, 2))
@given(array = arrays_of_positive_numbers())
def test_poisson_sigma(array, default):
    expected = np.where(array == 0, default, array**0.5)
    assert poisson_sigma(array, default) == approx(expected)
