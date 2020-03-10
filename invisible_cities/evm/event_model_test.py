import numpy as np

from pytest import mark

from hypothesis             import given
from hypothesis.strategies  import just
from hypothesis.strategies  import lists
from hypothesis.strategies  import one_of
from hypothesis.strategies  import floats
from hypothesis.strategies  import integers
from hypothesis.strategies  import composite

from .. types.ic_types   import xy
from .       event_model import Event

from .       event_model import Cluster
from .       event_model import Hit
from .       event_model import Voxel
from .       event_model import HitCollection
from .       event_model import HitEnergy
from .       event_model import KrEvent


@composite
def sensor_params_input(draw):
    npmt   = draw(integers())
    pmtwl  = draw(integers())
    nsipm  = draw(integers())
    sipmwl = draw(integers())
    return npmt, pmtwl, nsipm, sipmwl


@composite
def event_input(draw):
    evt_no = draw(integers())
    time   = draw(floats  (allow_nan=False))
    return evt_no, time


@composite
def voxel_input(draw):
    x     = draw(floats(  1,   5))
    y     = draw(floats(-10,  10))
    z     = draw(floats(.01,  .5))
    E     = draw(floats( 50, 100))
    size  = np.array([draw(floats(1,2)), draw(floats(1,2)), draw(floats(1,2))])
    return x, y, z, E, size

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


@composite
def hits(draw):
    Q, x, y, xvar, yvar, nsipm                            = draw(cluster_input())
    peak_number, E, z, x_peak, y_peak, s2ec, track_id, ep = draw(    hit_input())

    c = Cluster(Q, xy(x,y), xy(xvar,yvar), nsipm)
    h = Hit(peak_number, c, z, E, xy(x_peak, y_peak), s2ec, track_id, ep)
    return h


@mark.parametrize("test_class",
                  (Event,
                   HitCollection,
                   KrEvent))
@given(event_input())
def test_event(test_class, event_pars):
    evt_no, time = event_pars
    evt =  test_class(*event_pars)

    assert evt.event == evt_no
    assert evt.time  == time


@given(cluster_input())
def test_cluster(ci):
    Q, x, y, xvar, yvar, nsipm = ci
    xrms = np.sqrt(xvar)
    yrms = np.sqrt(yvar)
    r, phi =  np.sqrt(x ** 2 + y ** 2), np.arctan2(y, x)
    xyar   = (x, y)
    varar  = (xvar, yvar)
    pos    = np.stack(([x], [y]), axis=1)
    c      = Cluster(Q, xy(x,y), xy(xvar,yvar), nsipm)

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


@given(cluster_input(), hit_input())
def test_hit(ci, hi):
    Q, x, y, xvar, yvar, nsipm                            = ci
    peak_number, E, z, x_peak, y_peak, s2ec, track_id, ep = hi
    xyz = x, y, z

    c = Cluster(Q, xy(x,y), xy(xvar,yvar), nsipm)
    h = Hit(peak_number, c, z, E, xy(x_peak, y_peak), s2ec, track_id, ep)

    assert h.peak_number == peak_number
    assert h.npeak       == peak_number

    np.isclose (h.X       , x       , rtol=1e-4)
    np.isclose (h.Y       , y       , rtol=1e-4)
    np.isclose (h.Z       , z       , rtol=1e-4)
    np.isclose (h.E       , E       , rtol=1e-4)
    np.isclose (h.Xpeak   , x_peak  , rtol=1e-4)
    np.isclose (h.Ypeak   , y_peak  , rtol=1e-4)
    np.allclose(h.XYZ     , xyz     , rtol=1e-4)
    np.allclose(h.pos     , xyz     , rtol=1e-4)
    np.allclose(h.Ec      , s2ec    , rtol=1e-4)
    np.allclose(h.track_id, track_id, rtol=1e-4)
    np.allclose(h.Ep      , ep      , rtol=1e-4)


@given(voxel_input())
def test_voxel(vi):
    x, y, z, E, size = vi
    xyz = x, y, z
    v = Voxel(x, y, z, E, size)

    np.allclose(v.XYZ, xyz, rtol=1e-4)
    np.allclose(v.pos, xyz, rtol=1e-4)
    np.isclose (v.E  , E  , rtol=1e-4)
    np.isclose (v.X  , x  , rtol=1e-4)
    np.isclose (v.Y  , y  , rtol=1e-4)
    np.isclose (v.Z  , z  , rtol=1e-4)


def test_hit_collection_empty():
    hc = HitCollection(-1, -1)
    assert hc.hits == []


@given(lists(hits()))
def test_hit_collection_nonempty(hits):
    hc = HitCollection(-1, -1, hits=hits)
    assert hc.hits == hits


def test_kr_event_attributes():
    evt =  KrEvent(-1, -1)

    for attr in ["nS1", "nS2"]:
        assert getattr(evt, attr) == -1

    for attr in ["S1w", "S1h", "S1e", "S1t",
                 "S2w", "S2h", "S2e", "S2t", "S2q",
                 "Nsipm", "DT", "Z",
                 "X", "Y", "R", "Phi",
                 "Xrms", "Yrms"]:
        assert getattr(evt, attr) == []
