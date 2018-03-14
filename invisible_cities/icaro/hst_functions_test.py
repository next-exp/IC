import numpy as np

from   numpy.testing import assert_equal
from   numpy.testing import assert_allclose

from . import hst_functions as hst


def test_resolution_no_errors():
    R, Rbb = hst.resolution([None, 1, 1])

    assert_equal(R  .uncertainty, 0)
    assert_equal(Rbb.uncertainty, 0)


def test_resolution_scaling():
    _, Rbb1 = hst.resolution([None, 1, 1], E_from = 1)
    _, Rbb2 = hst.resolution([None, 1, 1], E_from = 2)

    assert_allclose(Rbb1.value * 2**0.5, Rbb2.value)
