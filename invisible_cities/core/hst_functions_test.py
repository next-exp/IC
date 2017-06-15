import numpy as np
from   numpy.testing import assert_equal, assert_allclose

from hypothesis             import given
from hypothesis.strategies  import floats
from hypothesis.extra.numpy import arrays

from . import hst_functions as hst


@given(arrays(float, 10, floats(min_value=-1e12, max_value=1e12)))
def test_shift_to_bin_centers(x):
    x_shifted = hst.shift_to_bin_centers(x)
    truth     = [np.mean(x[i:i+2]) for i in range(x.size-1)]
    assert_allclose(x_shifted, truth, atol=1e-100)


def test_resolution_no_errors():
    R, Rbb = hst.resolution([None, 1, 1])

    assert_equal(R  .uncertainty, 0)
    assert_equal(Rbb.uncertainty, 0)


def test_resolution_scaling():
    _, Rbb1 = hst.resolution([None, 1, 1], E_from = 1)
    _, Rbb2 = hst.resolution([None, 1, 1], E_from = 2)

    assert_allclose(Rbb1.value * 2**0.5, Rbb2.value)