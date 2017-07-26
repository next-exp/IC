import numpy as np
from   numpy.testing import assert_equal, assert_allclose

from pytest import mark

from hypothesis             import given
from hypothesis.strategies  import composite
from hypothesis.strategies  import integers
from hypothesis.strategies  import floats
from hypothesis.extra.numpy import arrays

from . import hst_functions as hst


@composite
def data_3d(draw,
            size =  100,
            xmin = -100, xmax = 100,
            ymin = -100, ymax = 100,
            zmin =    0, zmax = 600):
    x = draw(arrays(float, size, floats(min_value=xmin, max_value=xmax)))
    y = draw(arrays(float, size, floats(min_value=ymin, max_value=ymax)))
    z = draw(arrays(float, size, floats(min_value=zmin, max_value=zmax)))
    return x, y, z


@given(arrays(float, 10, floats(min_value=-1e5, max_value=1e5)))
def test_shift_to_bin_centers(x):
    x_shifted = hst.shift_to_bin_centers(x)
    truth     = [np.mean(x[i:i+2]) for i in range(x.size-1)]
    assert_allclose(x_shifted, truth, rtol=1e-6, atol=1e-6)


def test_resolution_no_errors():
    R, Rbb = hst.resolution([None, 1, 1])

    assert_equal(R  .uncertainty, 0)
    assert_equal(Rbb.uncertainty, 0)


def test_resolution_scaling():
    _, Rbb1 = hst.resolution([None, 1, 1], E_from = 1)
    _, Rbb2 = hst.resolution([None, 1, 1], E_from = 2)

    assert_allclose(Rbb1.value * 2**0.5, Rbb2.value)

@mark.slow
@given(integers(min_value=1, max_value=10),
       integers(min_value=1, max_value=10),
       data_3d (size=10, xmax=0, ymin=0  ))
def test_hist2d_profile(nbinsx, nbinsy, xye):
    x, y, e = xye
    (xp, yp, ep, ee), (eh, xh, yh, _), _ = hst.hist2d_profile(x, y, e, nbinsx, nbinsy, (-100, 0), (0, 100))
    assert xp.size == xh.size            == nbinsx
    assert yp.size == yh.size            ==          nbinsy
    assert ep.size == eh.size == ee.size == nbinsx * nbinsy
