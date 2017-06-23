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
from .       event_model import _S12
from .       event_model import S1
from .       event_model import S2
from .       event_model import S2Si
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
def waveform_input(draw, min_size=1, max_size=100):
    size = draw(integers(min_size, max_size))
    t    = draw(arrays(float, size, floats(0.1, 1000.))); t.sort() # times in order!
    E    = draw(arrays(float, size, floats(0.1, 100.)))
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
        np.allclose (wf.t , t, rtol=1e-4)
        np.allclose (wf.E , E, rtol=1e-4)
        np.isclose (wf.total_energy , np.sum(E), rtol=1e-4)
        np.isclose (wf.height , np.max(E), rtol=1e-4)
        np.isclose (wf.tpeak , t[np.argmax(E)], rtol=1e-4)
        np.isclose (wf.width , t[-1] - t[0], rtol=1e-4)
        assert wf.number_of_samples == size


@given(waveform_input())
def test_s12c(wform):
    _, t1, E1 = wform
    _, t2, E2 = wform

    s12d = {0: [t1, E1], 1: [t2, E2] }
    s12 = _S12(s12d)

    assert s12.number_of_peaks == len(s12d)
    for i in range(s12.number_of_peaks):
        if s12.peak_waveform(i).good_waveform:
            np.allclose (s12.peak_waveform(i).t , s12d[i][0], rtol=1e-4)
            np.allclose (s12.peak_waveform(i).E , s12d[i][1], rtol=1e-4)


@given(integers(min_value=31, max_value=40)) # pick a random event, limits from KrMC_pmaps fixture in conftest.py
def test_s1_s2(KrMC_pmaps, evt_no):
    *_, (s1data, s2data, _) = KrMC_pmaps

    s1data = s1data[evt_no]
    s2data = s2data[evt_no]

    s1 = S1(s1data)
    s2 = S2(s2data)

    assert s1.number_of_peaks == len(s1data)
    assert s2.number_of_peaks == len(s2data)

    assert sorted(s1.peaks) == sorted(s1data)
    for peak_no, peak in s1data.items():
        assert s1.peak_waveform(peak_no) == Waveform(*peak)


@given(integers(min_value=31, max_value=40))
def test_s2si(KrMC_pmaps, evt_no):
    *_, (_, s2data, s2sidata) = KrMC_pmaps

    s2d   = s2data[evt_no]
    s2sid = s2sidata[evt_no]

    s2si = S2Si(s2d, s2sid)
    assert s2si.number_of_peaks == len(s2d)

    for peak_number in range(s2si.number_of_peaks):
        assert (s2si.number_of_sipms_in_peak(peak_number) ==
                len(s2sid[peak_number]))

        np.array_equal(np.array(s2si.sipms_in_peak(peak_number)),
                       np.array(s2sid[peak_number].keys()))

        for sipm_number in s2si.sipms_in_peak(peak_number):
            w   = s2si.sipm_waveform(peak_number, sipm_number)
            wzs = s2si.sipm_waveform_zs(peak_number, sipm_number)
            E   = s2sid[peak_number][sipm_number]
            t = s2si.peak_waveform(peak_number).t
            np.allclose(w.E , E)
            np.allclose(wzs.E , E[E>0])
            np.allclose(wzs.t , t[E>0])


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
