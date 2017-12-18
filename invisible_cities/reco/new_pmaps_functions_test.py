import numpy as np

from hypothesis            import given
from hypothesis.strategies import integers
from hypothesis.strategies import floats

from .. evm .new_pmaps_test import peaks
from .. evm .new_pmaps      import  PMTResponses
from .. evm .new_pmaps      import SiPMResponses
from .. core.testing_utils  import assert_SensorResponses_equality
from .. core.testing_utils  import assert_Peak_equality
from .                      import new_peak_functions  as pkf
from .                      import new_pmaps_functions as pmf


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
    pmt_r        =  PMTResponses(pk. pmts.ids, pk. pmts.sum_over_times[:, np.newaxis])
    sipm_r       = SiPMResponses(pk.sipms.ids, pk.sipms.sum_over_times[:, np.newaxis])
    expected_pk  = type(pk)(time_average, pmt_r, sipm_r)

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
     rebinned_pmt_wfs ) = pkf.rebin_times_and_waveforms(pk.times,
                                                        pk.pmts.all_waveforms,
                                                        rebin_factor)

    _, rebinned_sipm_wfs = pkf.rebin_times_and_waveforms(pk.times,
                                                         pk.sipms.all_waveforms,
                                                         rebin_factor)

    rebinned_pmts  =  PMTResponses(pk. pmts.ids, rebinned_pmt_wfs)
    rebinned_sipms = SiPMResponses(pk.sipms.ids, rebinned_sipm_wfs)
    expected_pk  = type(pk)(rebinned_times,
                            rebinned_pmts,
                            rebinned_sipms)
    assert_Peak_equality(rebinned_pk, expected_pk)
