from itertools import combinations

import numpy as np

from pytest import approx
from pytest import mark

from hypothesis            import given
from hypothesis            import settings
from hypothesis            import HealthCheck
from hypothesis.strategies import integers
from hypothesis.strategies import floats
from hypothesis.strategies import dictionaries
from hypothesis.strategies import lists

from .. evm .pmaps_test     import peaks
from .. evm .pmaps_test     import pmaps
from .. evm .pmaps          import S2
from .. evm .pmaps          import  PMTResponses
from .. evm .pmaps          import SiPMResponses
from .. core.testing_utils  import assert_SensorResponses_equality
from .. core.testing_utils  import assert_Peak_equality
from .                      import peak_functions  as pkf
from .                      import pmaps_functions as pmf


@given(peaks())
def test_rebin_peak_rebin_factor_1(pk):
    _, pk       = pk
    rebinned_pk = pmf.rebin_peak(pk, 1)
    expected_pk = pk
    assert_Peak_equality(rebinned_pk, expected_pk)


@given(peaks())
def test_rebin_peak_collapse(pk):
    _, pk        = pk
    rebin_factor = pk.times.size
    rebinned_pk  = pmf.rebin_peak(pk, rebin_factor)

    time_average = [np.average(pk.times, weights=pk.pmts.sum_over_sensors)]
    widths       = [pk.width]
    pmt_r        =  PMTResponses(pk. pmts.ids, pk. pmts.sum_over_times[:, np.newaxis])
    sipm_r       = SiPMResponses(pk.sipms.ids, pk.sipms.sum_over_times[:, np.newaxis])
    expected_pk  = type(pk)(time_average, widths, pmt_r, sipm_r)

    assert_Peak_equality(rebinned_pk, expected_pk)


@given(peaks())
def test_rebin_peak_empty_sipms(pk):
    _, pk = pk

    # force empty sipms
    pk.sipms    = SiPMResponses.build_empty_instance()
    rebinned_pk = pmf.rebin_peak(pk, 2)

    assert_SensorResponses_equality(rebinned_pk.sipms, pk.sipms)


@given(peaks(), integers(1, 10))
def test_rebin_peak_same_output_class(pk, rebin_factor):
    _, pk       = pk
    rebinned_pk = pmf.rebin_peak(pk, rebin_factor)
    assert type(rebinned_pk) is type(pk)


@given(peaks(), floats(0, 1))
def test_rebin_peak(pk, fraction):
    _, pk        = pk
    rebin_factor = min(2, int(fraction * pk.times.size))
    rebinned_pk  = pmf.rebin_peak(pk, rebin_factor)

    (rebinned_times,
     rebinned_widths,
     rebinned_pmt_wfs ) = pkf.rebin_times_and_waveforms(pk.times,
                                                        pk.bin_widths,
                                                        pk.pmts.all_waveforms,
                                                        rebin_factor)

    (_, _,
     rebinned_sipm_wfs) = pkf.rebin_times_and_waveforms(pk.times,
                                                        pk.bin_widths,
                                                        pk.sipms.all_waveforms,
                                                        rebin_factor)

    rebinned_pmts  =  PMTResponses(pk. pmts.ids, rebinned_pmt_wfs)
    rebinned_sipms = SiPMResponses(pk.sipms.ids, rebinned_sipm_wfs)
    expected_pk  = type(pk)(rebinned_times,
                            rebinned_widths,
                            rebinned_pmts,
                            rebinned_sipms)
    assert_Peak_equality(rebinned_pk, expected_pk)


@mark.parametrize("threshold", (4000, 10000))
@given(peaks(subtype=S2))
def test_rebin_peak_threshold(threshold, pk):
    _, pk = pk

    pk_eng      = pk.total_energy
    pk_char     = pk.total_charge

    rebinned_pk = pmf.rebin_peak(pk, threshold, pmf.RebinMethod.threshold)

    assert rebinned_pk.total_energy == approx(pk_eng)
    assert rebinned_pk.total_charge == approx(pk_char)
    if pk_eng < threshold:
        assert rebinned_pk.times.shape[0] == 1
    else:
        assert np.all(rebinned_pk.pmts.sum_over_sensors >= threshold)


@given(pmaps(), pmaps(), pmaps(), pmaps())
@settings(suppress_health_check=(HealthCheck.too_slow,))
def test_pmap_event_id_selection(pmap0, pmap1, pmap2, pmap3):
    pmaps  = {0:pmap0, 12:pmap1, 345:pmap2, 6789:pmap3}
    events = list(pmaps)
    for chosen in combinations(events, 2):
        pm0, pm1 = chosen
        filtered_pmaps = pmf.pmap_event_id_selection(pmaps, chosen)
        assert len(filtered_pmaps) == 2
        assert pm0 in filtered_pmaps and pm1 in filtered_pmaps
        assert filtered_pmaps[pm0] == pmaps[pm0]
        assert filtered_pmaps[pm1] == pmaps[pm1]
