import numpy as np

from numpy.testing import assert_equal
from pytest        import mark

from hypothesis             import given
from hypothesis.strategies  import floats
from hypothesis.strategies  import integers
from hypothesis.strategies  import composite
from hypothesis.extra.numpy import arrays

from .. core.ic_types    import xy
from .. core.ic_types    import minmax
from .       event_model import SensorParams
from .       event_model import Event
from .       event_model import Waveform
from .       event_model import S12
from .       event_model import Cluster
from .       event_model import Hit
from .       event_model import HitCollection
from .       event_model import PersistentHitCollection
from .       event_model import KrEvent
from .       event_model import PersistentKrEvent


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
def waveform_input(draw, min_size=1, max_size=1000):
    size = draw(integers(min_size, max_size))
    t    = draw(arrays(float, size, floats())); t.sort() # times in order!
    E    = draw(arrays(float, size, floats()))
    return size, t, E


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


@given(sensor_params_input())
def test_sensor_params(sensor_pars):
    npmt, pmtwl, nsipm, sipmwl = sensor_pars
    sp =  SensorParams(*sensor_pars)

    assert sp.npmt   == sp.NPMT   == npmt
    assert sp.nsipm  == sp.NSIPM  == nsipm
    assert sp.pmtwl  == sp.PMTWL  == pmtwl
    assert sp.sipmwl == sp.SIPMWL == sipmwl


@mark.parametrize("test_class",
                  (Event,
                   HitCollection,
                   PersistentHitCollection,
                   KrEvent,
                   PersistentKrEvent))
@given(event_input())
def test_event(test_class, event_pars):
    evt_no, time = event_pars
    evt =  test_class(*event_pars)

    assert evt.event == evt_no
    assert evt.time  == time


@given(waveform_input())
def test_waveform(waveform_pars):
    size, t, E = waveform_pars
    wf =  Waveform(t, E)
    if np.array_equal(wf.t, np.array([0])) or np.array_equal(wf.E,np.array([0])):
         assert wf.good_waveform == False
    else:
        assert wf.good_waveform == True

    if wf.good_waveform == True:
        assert_equal(wf.t, t)
        assert_equal(wf.E, E)
        assert wf == wf
        assert wf.total_energy      == np.sum(E)
        assert wf.height            == np.max(E)
        assert wf.number_of_samples == size
        assert wf.tpeak             == t[np.argmax(E)]
        assert wf.tmin_tmax         == minmax(t[0], t[-1])
        assert wf.width             == t[-1] - t[0]


@given(integers(min_value=31, max_value=40)) # pick a random event, limits from KrMC_pmaps fixture in conftest.py
def test_s12(KrMC_pmaps, evt_no):
    *_, (s1data, s2data, _) = KrMC_pmaps

    s1data = s1data[evt_no]
    s2data = s2data[evt_no]

    s1 = S12(s1data, "S1")
    s2 = S12(s2data, "S2")

    assert s1.type == "S1"
    assert s2.type == "S2"

    assert s1.number_of_peaks == len(s1data)
    assert s2.number_of_peaks == len(s2data)

    assert sorted(s1.peaks) == sorted(s1data)
    for peak_no, peak in s1data.items():
        assert s1.peak_waveform(peak_no) == Waveform(*peak)


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


@mark.parametrize("test_class",
                  (HitCollection,
                   PersistentHitCollection))
def test_hit_collection_empty(test_class):
    hc = test_class(-1, -1)
    assert hc.hits == []


@mark.parametrize("test_class",
                  (KrEvent,
                   PersistentKrEvent))
def test_kr_event_attributes(test_class):
    evt =  test_class(-1, -1)

    for attr in ["nS1", "nS2"]:
        assert getattr(evt, attr) == -1

    for attr in ["S1w", "S1h", "S1e", "S1t",
                 "S2w", "S2h", "S2e", "S2t", "S2q",
                 "Nsipm", "DT", "Z",
                 "X", "Y", "R", "Phi",
                 "Xrms", "Yrms"]:
        assert getattr(evt, attr) == []
