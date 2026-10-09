import numpy as np

from . ic_types          import minmax

from pytest import raises

from hypothesis            import given
from hypothesis.strategies import floats
from hypothesis.strategies import builds


def make_minmax(a,b):
    # Ensure that arguments respect the required order
    if a > b: a, b = b, a
    return minmax(a, b)

sensible_floats = floats(min_value=0.5, max_value=1e3, allow_nan=False, allow_infinity=False)
minmaxes        = builds(make_minmax, sensible_floats, sensible_floats)

@given(sensible_floats, sensible_floats)
def test_minmax_interval(a,b):
    if a > b: a, b = b, a
    assert np.allclose(minmax(a,b).interval , (a,b), rtol=1e-4)

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
def test_minmax_add(mm, f):
    lo, hi = mm
    raised = mm + f
    np.isclose (raised.min , lo + f, rtol=1e-4)
    np.isclose (raised.max , hi + f, rtol=1e-4)


@given(minmaxes, sensible_floats)
def test_minmax_mul(mm, f):
    lo, hi = mm
    scaled = mm * f
    np.isclose (scaled.min , lo * f, rtol=1e-4)
    np.isclose (scaled.max , hi * f, rtol=1e-4)


@given(minmaxes, sensible_floats)
def test_minmax_div(mm, f):
    lo, hi = mm
    scaled = mm / f
    np.isclose (scaled.min , lo / f, rtol=1e-4)
    np.isclose (scaled.max , hi / f, rtol=1e-4)


@given(minmaxes, sensible_floats)
def test_minmax_sub(mm, f):
    lo, hi = mm
    lowered = mm - f
    np.isclose (lowered.min , lo - f, rtol=1e-4)
    np.isclose (lowered.max , hi - f, rtol=1e-4)
