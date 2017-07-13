import numpy as np

from . ic_types          import minmax
from . ic_types          import xy
from . ic_types_c        import xy as cxy
from . ic_types_c        import minmax as cmm

from pytest import raises

from hypothesis            import given
from hypothesis.strategies import floats
from hypothesis.strategies import builds


def make_minmax(a,b):
    # Ensure that arguments respect the required order
    if a > b: a, b = b, a
    return minmax(a, b)

def make_xy(a,b):
    return xy(a,b)

def make_cminmax(a,b):
    # Ensure that arguments respect the required order
    if a > b: a, b = b, a
    return cmm(a, b)

def make_xyc(a,b):
    return cxy(a,b)


sensible_floats = floats(min_value=0.5, max_value=1e3, allow_nan=False, allow_infinity=False)
minmaxes        = builds(make_minmax, sensible_floats, sensible_floats)
cmms            = builds(make_cminmax, sensible_floats, sensible_floats)
xys             = builds(make_xy, sensible_floats, sensible_floats)
cxys            = builds(make_xyc, sensible_floats, sensible_floats)


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


@given(cmms, sensible_floats)
def test_cminmax_add(mm, f):
    lo, hi = mm
    raised = mm + f
    np.isclose (raised.min , lo + f, rtol=1e-4)
    np.isclose (raised.max , hi + f, rtol=1e-4)


@given(cmms, sensible_floats)
def test_cminmax_div(mm, f):
    lo, hi = mm
    scaled = mm / f
    np.isclose (scaled.min , lo / f, rtol=1e-4)
    np.isclose (scaled.max , hi / f, rtol=1e-4)


@given(cmms, sensible_floats)
def test_cminmax_mul(mm, f):
    lo, hi = mm
    scaled = mm * f
    np.isclose (scaled.min , lo * f, rtol=1e-4)
    np.isclose (scaled.max , hi * f, rtol=1e-4)


@given(cmms, sensible_floats)
def test_cminmax_sub(mm, f):
    lo, hi = mm
    lowered = mm - f
    np.isclose (lowered.min , lo - f, rtol=1e-4)
    np.isclose (lowered.max , hi - f, rtol=1e-4)


@given(xys, sensible_floats, sensible_floats)
def test_xy(xy, a, b):
    ab  = a, b
    r   = np.sqrt(a ** 2 + b ** 2)
    phi = np.arctan2(b, a)
    pos = np.stack(([a], [b]), axis=1)
    np.isclose (xy.x  ,   a, rtol=1e-4)
    np.isclose (xy.y  ,   b, rtol=1e-4)
    np.isclose (xy.X  ,   a, rtol=1e-4)
    np.isclose (xy.Y  ,   b, rtol=1e-4)
    np.isclose (xy.XY ,  ab, rtol=1e-4)
    np.isclose (xy.R  ,   r, rtol=1e-4)
    np.isclose (xy.Phi, phi, rtol=1e-4)
    np.allclose(xy.pos, pos, rtol=1e-3, atol=1e-03)


@given(cxys, sensible_floats, sensible_floats)
def test_cxy(xyc, a, b):
    ab  = a, b
    r   = np.sqrt(a ** 2 + b ** 2)
    phi = np.arctan2(b, a)
    pos = np.stack(([a], [b]), axis=1)
    np.isclose (xyc.x  ,   a, rtol=1e-4)
    np.isclose (xyc.y  ,   b, rtol=1e-4)
    np.isclose (xyc.X  ,   a, rtol=1e-4)
    np.isclose (xyc.Y  ,   b, rtol=1e-4)
    np.isclose (xyc.XY ,  ab, rtol=1e-4)
    np.isclose (xyc.R  ,   r, rtol=1e-4)
    np.isclose (xyc.Phi, phi, rtol=1e-4)
    np.allclose(xyc.pos, pos, rtol=1e-3, atol=1e-03)
