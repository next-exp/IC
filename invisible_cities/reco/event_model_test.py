import numpy as np

from hypothesis             import given
from hypothesis.strategies  import floats
from hypothesis.strategies  import integers
from hypothesis.strategies  import composite

from .. core.ic_types    import xy
from .       event_model import Cluster
from .       event_model import Hit


@composite
def cluster_input(draw, min_value=0, max_value=100):
    x     = draw(floats  (  1,   5))
    y     = draw(floats  (-10,  10))
    xrms  = draw(floats  (.01,  .5))
    yrms  = draw(floats  (.10,  .9))
    Q     = draw(floats  ( 50, 100))
    nsipm = draw(integers(  1,  20))
    return Q, x, y, xrms, yrms, nsipm


@composite
def hit_input(draw, min_value=0, max_value=100):
    z           = draw(floats  (.1,  .9))
    s2_energy   = draw(floats  (50, 100))
    peak_number = draw(integers( 1,  20))
    return peak_number, s2_energy, z


@given(cluster_input(1))
def test_cluster(ci):
    Q, x, y, xrms, yrms, nsipm = ci
    r, phi =  np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)
    xyar   = x, y
    rmsar  = xrms, yrms
    pos    = np.stack(([x], [y]), axis=1)
    c      = Cluster(Q, xy(x,y), xy(xrms,yrms), nsipm)

    assert c.nsipm == nsipm
    np.isclose (c.Q   ,     Q, rtol=1e-4)
    np.isclose (c.X   ,     x, rtol=1e-4)
    np.isclose (c.Y   ,     y, rtol=1e-4)
    np.isclose (c.Xrms,  xrms, rtol=1e-4)
    np.isclose (c.Yrms,  yrms, rtol=1e-4)
    np.isclose (c.rms , rmsar, rtol=1e-4)
    np.allclose(c.XY  ,  xyar, rtol=1e-4)
    np.isclose (c.R   ,     r, rtol=1e-4)
    np.isclose (c.Phi ,   phi, rtol=1e-4)
    np.allclose(c.pos ,   pos, rtol=1e-4)


@given(cluster_input(1), hit_input(1))
def test_hit(ci, hi):
    Q, x, y, xrms, yrms, nsipm = ci
    peak_number, E, z          = hi
    xyz = x, y, z

    c = Cluster(Q, xy(x,y), xy(xrms,yrms), nsipm)
    h = Hit(peak_number, c, z, E)

    assert h.peak_number == peak_number
    assert h.npeak       == peak_number
    np.isclose (h.z        ,   z, rtol=1e-4)
    np.isclose (h.Z        ,   z, rtol=1e-4)
    np.isclose (h.s2_energy,   E, rtol=1e-4)
    np.isclose (h.E        ,   E, rtol=1e-4)
    np.allclose(h.XYZ      , xyz, rtol=1e-4)
    np.allclose(h.VXYZ     , xyz, rtol=1e-4)
