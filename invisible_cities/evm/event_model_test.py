import numpy as np

from pytest import mark

from hypothesis             import given
from hypothesis.strategies  import just
from hypothesis.strategies  import one_of
from hypothesis.strategies  import floats
from hypothesis.strategies  import integers
from hypothesis.strategies  import composite

from .. types.ic_types   import xy
from .. types.symbols    import HitEnergy

from .       event_model import Cluster

@composite
def cluster_input(draw):
    x     = draw(floats  (  1,   5))
    y     = draw(floats  (-10,  10))
    xvar  = draw(floats  (.01,  .5))
    yvar  = draw(floats  (.10,  .9))
    Q     = draw(floats  ( 50, 100))
    nsipm = draw(integers(  1,  20))
    return Q, x, y, xvar, yvar, nsipm


@composite
def hit_input(draw):
    z           = draw(floats  (.1,  .9))
    s2_energy   = draw(floats  (50, 100))
    peak_number = draw(integers( 1,  20))
    x_peak      = draw(floats (-10., 2.))
    y_peak      = draw(floats (-20., 5.))
    s2_energy_c = draw(one_of(just(-1), floats  (50, 100)))
    track_id    = draw(one_of(just(-1), integers( 0,  10)))
    Ep          = draw(one_of(just(-1), floats  (50, 100)))
    return peak_number, s2_energy, z, x_peak, y_peak, s2_energy_c, track_id, Ep


@given(cluster_input())
def test_cluster(ci):
    Q, x, y, xvar, yvar, nsipm = ci
    xrms = np.sqrt(xvar)
    yrms = np.sqrt(yvar)
    r, phi =  np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)
    xyar   = (x, y)
    varar  = (xvar, yvar)
    pos    = np.stack(([x], [y]), axis=1)
    c      = Cluster(Q, xy(x,y), xy(xvar,yvar), nsipm, z=None)

    assert c.nsipm == nsipm
    np.isclose (c.Q     , Q    , rtol=1e-4)
    np.isclose (c.X     , x    , rtol=1e-4)
    np.isclose (c.Y     , y    , rtol=1e-4)
    np.isclose (c.Xrms  , xrms , rtol=1e-4)
    np.isclose (c.Yrms  , yrms , rtol=1e-4)
    np.isclose (c.var.XY, varar, rtol=1e-4)
    np.allclose(c.XY    , xyar , rtol=1e-4)
    np.isclose (c.R     , r    , rtol=1e-4)
    np.isclose (c.Phi   , phi  , rtol=1e-4)
    np.allclose(c.posxy , pos  , rtol=1e-4)


@mark.parametrize("value", "E Ec Ep".split())
def test_hitenergy_value(value):
    assert getattr(HitEnergy, value).value == value
