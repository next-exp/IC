import numpy as np

from numpy.testing import assert_equal
from pytest        import mark

from hypothesis             import given
from hypothesis.strategies  import floats
from hypothesis.strategies  import integers
from hypothesis.strategies  import composite
from hypothesis.extra.numpy import arrays

from .. core.ic_types_c    import xy
from .. core.ic_types_c    import minmax

from .  pmaps import S12
from .  pmaps import S1
from .  pmaps import S2
from .  pmaps import S2Si
from .  pmaps import Peak


@composite
def peak_input(draw, min_size=1, max_size=100):
    size = draw(integers(min_size, max_size))
    t    = draw(arrays(float, size, floats(0.1, 10000.))); t.sort() # times in order!
    E    = draw(arrays(float, size, floats(-10., 1000.)))
    return size, t, E




@given(peak_input())
def test_waveform(waveform_pars):
    size, t, E = waveform_pars
    wf =  Peak(t, E)

    np.allclose (wf.t , t, rtol=1e-4)
    np.allclose (wf.E , E, rtol=1e-4)
    np.isclose (wf.total_energy , np.sum(E), rtol=1e-4)
    np.isclose (wf.height , np.max(E), rtol=1e-4)
    np.isclose (wf.tpeak , t[np.argmax(E)], rtol=1e-4)
    np.isclose (wf.width , t[-1] - t[0], rtol=1e-4)
    assert wf.number_of_samples == size


@given(peak_input())
def test_s12c(wform):
    _, t1, E1 = wform
    _, t2, E2 = wform

    s12d = {0: [t1, E1], 1: [t2, E2] }
    s12 = S12(s12d)

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
    for peak_no, (t,E) in s1data.items():
        pwf = s1.peak_waveform(peak_no)
        pk =  Peak(t,E)
        np.allclose (pwf.t , pk.t, rtol=1e-4)
        np.allclose (pwf.E , pk.E, rtol=1e-4)


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
