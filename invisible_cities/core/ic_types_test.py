import numpy as np

from . ic_types          import minmax
from . ic_types          import xy

from pytest import raises

from hypothesis            import given
from hypothesis.strategies import floats
from hypothesis.strategies import builds

sensible_floats = floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False)

def make_minmax(a,b):
    # Ensure that arguments respect the required order
    if a > b: a,b = b,a
    return minmax(a,b)

minmaxes = builds(make_minmax, sensible_floats, sensible_floats)

@given(sensible_floats, sensible_floats)
def test_minmax_does_not_accept_min_greater_than_max(a,b):
    # minmax defines a semi-open interval, so equality of limits is
    # acceptable, but min > max is not.
    if a <= b:
        minmax(a,b)
    else:
        with raises(AssertionError):
            minmax(a,b)

@given(minmaxes, sensible_floats)
def test_minmax_mul(mm, f):
    lo, hi = mm
    scaled = mm * f
    assert scaled.min == lo * f
    assert scaled.max == hi * f

@given(minmaxes, sensible_floats)
def test_minmax_sub(mm, f):
    lo, hi = mm
    lowered = mm - f
    assert lowered.min == lo - f
    assert lowered.max == hi - f

@given(minmaxes, sensible_floats)
def test_minmax_add(mm, f):
    lo, hi = mm
    raised = mm + f
    assert raised.min == lo + f
    assert raised.max == hi + f


def make_xy(a,b):
    return xy(a,b)

xys = builds(make_xy, sensible_floats, sensible_floats)

@given(xys,  sensible_floats, sensible_floats)
def test_xy(xy, a, b):
    #xy, a, b = xys
    np.isclose(xy.x, a, rtol=1e-4)
    np.isclose(xy.y, b, rtol=1e-4)
    np.isclose(xy.X, a, rtol=1e-4)
    np.isclose(xy.Y, b, rtol=1e-4)
    np.isclose(np.array(xy.XY), np.array((a,b)), rtol=1e-4)
    np.isclose(xy.R, np.sqrt(a ** 2 + b ** 2), rtol=1e-4)
    np.isclose(xy.Phi, np.arctan2(b, a), rtol=1e-4)
    assert np.allclose(xy.pos, np.stack(([a], [b]), axis=1),
                       rtol=1e-3, atol=1e-03)
