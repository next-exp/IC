import numpy as np

from pytest import approx
from pytest import mark
from pytest import fixture

from hypothesis               import given
from hypothesis               import assume
from hypothesis.strategies    import composite
from hypothesis.strategies    import floats
from hypothesis.strategies    import integers
from hypothesis.extra.numpy   import arrays

from ..core.testing_utils     import exactly
from ..core.testing_utils     import previous_float
from ..core.testing_utils     import assert_Peak_equality
from ..core.testing_utils     import assert_PMap_equality
from ..core.system_of_units_c import units
from ..core.fit_functions     import gauss
from ..evm .new_pmaps         import PMTResponses
from ..evm .new_pmaps         import SiPMResponses
from ..evm .new_pmaps         import S1
from ..evm .new_pmaps         import S2
from ..evm .new_pmaps         import PMap
from ..types.ic_types_c       import minmax
from .                        import new_peak_functions as pf


wf_min =   0
wf_max = 100


@composite
def waveforms(draw):
    n_samples = draw(integers(1, 50))
    return draw(arrays(float, n_samples, floats(wf_min, wf_max)))


@composite
def multiple_waveforms(draw):
    n_sensors = draw(integers(1, 10))
    n_samples = draw(integers(1, 50))
    return draw(arrays(float, (n_sensors, n_samples), floats(wf_min, wf_max)))


@composite
def times_and_waveforms(draw):
    waveforms = draw(multiple_waveforms())
    n_samples = waveforms.shape[1]
    times     = draw(arrays(float, n_samples, floats(0, 10*n_samples), unique=True))
    return times, waveforms


@composite
def rebinned_sliced_waveforms(draw):
    times, wfs = draw(times_and_waveforms())
    assume(times.size >= 5)

    indices     = np.arange(times.size)
    first       = draw(integers(        0, times.size - 2))
    last        = draw(integers(first + 1, times.size - 1))
    slice_      = slice(first, last + 1)
    indices     = indices[   slice_]
    times_slice = times  [   slice_]
    wfs_slice   = wfs    [:, slice_]
    rebin       = draw(integers(1, 5))
    (times_slice,
     wfs_slice) = pf.rebin_times_and_waveforms(times_slice, wfs_slice, rebin)
    return times, wfs, indices, times_slice, wfs_slice, rebin


@composite
def peak_indices(draw):
    size    = draw(integers(10, 50))
    indices = draw(arrays(int, size, integers(0, 5 * size), unique=True))
    indices = np.sort(indices)
    stride  = draw(integers(1, 5))
    peaks   = np.split(indices, 1 + np.where(np.diff(indices) > stride)[0])
    return indices, peaks, stride


@fixture
def wf_with_indices(n_sensors=5, n_samples=500, length=None, first=None):
    times = np.arange(n_samples) * 25 * units.ns
    wfs   = np.zeros((n_sensors, n_samples))
    amps  = np.random.uniform(50, 100, size=(n_sensors, 1))
    if length is None:
        length = np.random.randint(2, n_samples // 2)
    if first  is None:
        first  = np.random.randint(0, n_samples // 5)
    indices = np.arange(first, first + length)
    x_eval  = np.linspace(-3, 3, length)
    wfs[:, indices] = gauss(x_eval, amps, 0, 1)
    return times, wfs, indices


@fixture
def pmt_and_sipm_wfs_with_indices(n_pmt=3, n_sipm=10, n_samples_sipm=10):
    n_samples_pmt = n_samples_sipm * 40
    first_pmt     = np.random.randint(0, n_samples_pmt // 5)
    length_pmt    = np.random.randint(2, n_samples_pmt // 2)
    first_sipm    = first_pmt // 40
    length_sipm   = int(np.ceil((first_pmt + length_pmt) / 40)) - first_sipm
    times,  pmt_wfs,  pmt_indices = wf_with_indices(n_pmt , n_samples_sipm * 40,
                                                    length_pmt, first_pmt)

    _    , sipm_wfs, sipm_indices = wf_with_indices(n_sipm, n_samples_sipm,
                                                    length_sipm, first_sipm)
    return times, pmt_wfs, sipm_wfs, pmt_indices, sipm_indices


@fixture
def s1_and_s2_with_indices(n_pmt=3, n_sipm=10, n_samples_sipm=40):
    n_samples_pmt_s1 = 400
    length_pmt_s1    = np.random.randint(5, 20)

    times_s1, pmt_wfs_s1, pmt_indices_s1 = wf_with_indices(n_pmt,
                                                           n_samples_pmt_s1,
                                                           length_pmt_s1)
    sipm_wfs_s1 = np.zeros((n_sipm, n_samples_pmt_s1 // 40))

    n_samples_pmt_s2 = n_samples_sipm * 40
    first_pmt        = np.random.randint( 0, n_samples_pmt_s2 // 5)
    length_pmt_s2    = np.random.randint(40, n_samples_pmt_s2 // 2)
    times_s2, pmt_wfs_s2, pmt_indices_s2 = wf_with_indices(n_pmt,
                                                           n_samples_pmt_s2,
                                                           length_pmt_s2,
                                                           first_pmt)

    first_sipm  = first_pmt // 40
    length_sipm = int(np.ceil((first_pmt + length_pmt_s2) / 40)) - first_sipm
    _    , sipm_wfs_s2, sipm_indices = wf_with_indices(n_sipm, n_samples_sipm,
                                                       length_sipm, first_sipm)

    times_s2 += times_s1[-1] + np.diff(times_s1)[-1]
    times     = np.concatenate([   times_s1,    times_s2]        )
    pmt_wfs   = np.concatenate([ pmt_wfs_s1,  pmt_wfs_s2], axis=1)
    sipm_wfs  = np.concatenate([sipm_wfs_s1, sipm_wfs_s2], axis=1)

    pmt_indices_s2 += n_samples_pmt_s1
    sipm_indices   += n_samples_pmt_s1 // 40

    s1_params = {
    "time"        : minmax(times_s1[0], times_s1[-1]),
    "length"      : minmax(5, 20),
    "stride"      : 1,
    "rebin_stride": 1}

    s2_params = {
    "time"        : minmax(times_s2[0], times_s2[-1]),
    "length"      : minmax(40, n_samples_pmt_s2 // 2),
    "stride"      :  1,
    "rebin_stride": 40}

    return (times, pmt_wfs, sipm_wfs,
            pmt_indices_s1, pmt_indices_s2, sipm_indices,
            s1_params, s2_params)


@given(waveforms())
def test_indices_and_wf_above_threshold_minus_inf(wf):
    thr = -np.inf
    indices, wf_above_thr = pf.indices_and_wf_above_threshold(wf, thr)
    assert      indices == exactly(np.arange(wf.size))
    assert wf_above_thr == approx (wf)


@given(waveforms())
def test_indices_and_wf_above_threshold_min(wf):
    thr = previous_float(np.min(wf))
    indices, wf_above_thr = pf.indices_and_wf_above_threshold(wf, thr)
    assert      indices == exactly(np.arange(wf.size))
    assert wf_above_thr == approx (wf)


@given(waveforms())
def test_indices_and_wf_above_threshold_plus_inf(wf):
    thr = +np.inf
    indices, wf_above_thr = pf.indices_and_wf_above_threshold(wf, thr)
    assert np.size(indices)      == 0
    assert np.size(wf_above_thr) == 0


@given(waveforms())
def test_indices_and_wf_above_threshold_max(wf):
    thr = np.max(wf)
    indices, wf_above_thr = pf.indices_and_wf_above_threshold(wf, thr)

    assert indices     .size == 0
    assert wf_above_thr.size == 0


@given(waveforms(), floats(wf_min, wf_max))
def test_indices_and_wf_above_threshold(wf, thr):
    indices, wf_above_thr = pf.indices_and_wf_above_threshold(wf, thr)
    expected_indices      = np.where(wf > thr)[0]
    expected_wf           = wf[expected_indices]
    assert      indices == exactly(expected_indices)
    assert wf_above_thr == approx (expected_wf)


@given(multiple_waveforms())
def test_select_wfs_above_time_integrated_thr_minus_inf(sipm_wfs):
    thr      = -np.inf
    ids, wfs = pf.select_wfs_above_time_integrated_thr(sipm_wfs, thr)

    assert ids == exactly(np.arange(sipm_wfs.shape[0]))
    assert wfs == exactly(sipm_wfs)


@given(multiple_waveforms())
def test_select_wfs_above_time_integrated_thr_plus_inf(sipm_wfs):
    thr      = +np.inf
    ids, wfs = pf.select_wfs_above_time_integrated_thr(sipm_wfs, thr)

    assert ids.size == 0
    assert wfs.size == 0


@given(multiple_waveforms(), floats(wf_min, wf_max))
def test_select_wfs_above_time_integrated_thr(sipm_wfs, thr):
    ids, wfs     = pf.select_wfs_above_time_integrated_thr(sipm_wfs, thr)
    expected_ids = np.where(np.sum(sipm_wfs, axis=1) >= thr)[0]
    expected_wfs = sipm_wfs[expected_ids]

    assert ids == exactly(expected_ids)
    assert wfs == approx (expected_wfs)


@given(peak_indices())
def test_split_in_peaks(peak_data):
    indices, expected_peaks, stride = peak_data
    peaks = pf.split_in_peaks(indices, stride)

    assert len(peaks) == len(expected_peaks)
    for got, expected in zip(peaks, expected_peaks):
        assert got == exactly(expected)


@given(peak_indices())
def test_select_peaks_without_bounds(peak_data):
    _, peaks , _ = peak_data
    t_limits = minmax(-np.inf, np.inf)
    l_limits = minmax(-np.inf, np.inf)
    selected = pf.select_peaks(peaks, t_limits, l_limits)
    assert len(selected) == len(peaks)
    for got, expected in zip(selected, peaks):
        assert got == exactly(expected)


@given(peak_indices(),
       floats(0,  5), floats( 6, 10),
       floats(0, 10), floats(11, 20))
def test_select_peaks_filtered_out(peak_data, i0, i1, l0, l1):
    _, peaks , _ = peak_data
    i_limits = minmax(i0, i1)
    l_limits = minmax(l0, l1)
    selected = pf.select_peaks(peaks, i_limits * 25 * units.ns, l_limits)
    for peak in selected:
        assert i0 <=                peak[0] <= i1
        assert l0 <= peak[-1] + 1 - peak[0] <= l1


def test_select_peaks_right_length_with_holes():
    peak_with_hole = np.array([1, 2, 3,   7, 8, 9])
    length_correct = peak_with_hole[-1] + 1 - peak_with_hole[0]
    length_wrong   = len(peak_with_hole)

    l_limits_correct = minmax(length_correct - 1, length_correct + 1)
    l_limits_wrong   = minmax(length_wrong   - 1, length_wrong   + 1)

    peaks    = (peak_with_hole,)
    i_limits = minmax(0, 10) * 25 * units.ns

    selected_correct = pf.select_peaks(peaks, i_limits, l_limits_correct)
    selected_wrong   = pf.select_peaks(peaks, i_limits, l_limits_wrong)

    assert selected_correct[0] == exactly(peak_with_hole)
    assert selected_wrong      == ()


@given(peak_indices(),
       floats(0,  5), floats( 6, 10),
       floats(0, 10), floats(11, 20))
def test_select_peaks(peak_data, t0, t1, l0, l1):
    _, peaks , _ = peak_data
    i_limits = minmax(t0, t1)
    t_limits = i_limits * 25 * units.ns
    l_limits = minmax(l0, l1)
    selected = pf.select_peaks(peaks, t_limits, l_limits)

    select = lambda ids: (i_limits.contains(ids[ 0]) and
                          i_limits.contains(ids[-1]) and
                          l_limits.contains(ids[-1] + 1 - ids[0]))
    expected_peaks = tuple(filter(select, peaks))

    assert len(selected) == len(expected_peaks)
    for got, expected in zip(selected, expected_peaks):
        assert got == exactly(expected)


@given(rebinned_sliced_waveforms())
def test_pick_slice_and_rebin(wfs_slice_data):
    times, wfs, indices, times_slice, wfs_slice, rebin = wfs_slice_data
    sliced_times, sliced_wfs = pf.pick_slice_and_rebin(indices, times,
                                                       wfs, rebin)

    assert sliced_times == approx(times_slice)
    assert sliced_wfs   == approx(  wfs_slice)


def test_build_pmt_responses(wf_with_indices):
    times, wfs, indices = wf_with_indices
    ids = np.arange(wfs.shape[0])
    ts, pmt_r = pf.build_pmt_responses(indices, times,
                                       wfs, ids, 1, False)
    assert ts                  == approx (times[indices])
    assert pmt_r.ids           == exactly(ids)
    assert pmt_r.all_waveforms == approx (wfs[:, indices])


def test_build_sipm_responses(wf_with_indices):
    times, wfs, indices = wf_with_indices
    ids = np.arange(wfs.shape[0])
    wfs_slice       = wfs[:, indices]
    peak_integrals  = wfs_slice.sum(axis=1)
    below_thr_index = np.argmin (peak_integrals)
    # next_float doesn't work here
    thr             = peak_integrals[below_thr_index] * 1.000001
    sipm_r          = pf.build_sipm_responses(indices, times, wfs, 1, thr)

    expected_ids = np.delete(      ids, below_thr_index)
    expected_wfs = np.delete(wfs_slice, below_thr_index, axis=0)
    assert sipm_r.ids           == exactly(expected_ids)
    assert sipm_r.all_waveforms == approx (expected_wfs)


@mark.parametrize("Pk rebin with_sipms".split(),
                  ((S1,  1, False),
                   (S2,  1, False),
                   (S2, 40, True )))
def test_build_peak_development(pmt_and_sipm_wfs_with_indices,
                                Pk, rebin, with_sipms):
    (times, pmt_wfs, sipm_wfs,
     pmt_indices, sipm_indices) = pmt_and_sipm_wfs_with_indices
    pmt_ids  = np.arange( pmt_wfs.shape[0])

    if with_sipms:
        sipm_ids = np.arange(sipm_wfs.shape[0])
        rebin    = 40
        indices  = sipm_indices
        sipm_r = SiPMResponses(sipm_ids, sipm_wfs[:, indices])
    else:
        rebin   = 1
        indices = pmt_indices
        sipm_r = SiPMResponses.build_empty_instance()

    (rebinned_times,
     rebinned_wfs  ) = pf.rebin_times_and_waveforms(times, pmt_wfs, rebin)
    pmt_r            = PMTResponses(pmt_ids, rebinned_wfs[:, indices])
    expected_peak    = S2(rebinned_times[indices], pmt_r, sipm_r)

    peak = pf.build_peak(pmt_indices, times,
                         pmt_wfs, pmt_ids,
                         rebin_stride = rebin,
                         with_sipms   = with_sipms,
                         Pk           = Pk,
                         sipm_wfs     = sipm_wfs,
                         thr_sipm_s2  = -1)

    assert_Peak_equality(peak, expected_peak)


def test_find_peaks_trigger_style(pmt_and_sipm_wfs_with_indices):
    (times, pmt_wfs, sipm_wfs,
     pmt_indices, sipm_indices) = pmt_and_sipm_wfs_with_indices

    t_slice      = times[pmt_indices]
    time_range   = minmax(      t_slice[0] - 1,      t_slice[-1] + 2)
    length_range = minmax(pmt_indices.size - 1, pmt_indices.size + 2)
    stride       =   1
    rebin_stride =  40
    pmt_ids      = [-1]

    wf    = pmt_wfs[0]
    wfs   = wf[np.newaxis]
    peaks = pf.find_peaks(wf, pmt_indices,
                          time_range, length_range,
                          stride, rebin_stride,
                          S2, pmt_ids)

    (rebinned_times,
     rebinned_wfs  ) = pf.rebin_times_and_waveforms(times [pmt_indices],
                                                    wfs[:, pmt_indices],
                                                    rebin_stride)

    pmt_r            =  PMTResponses(pmt_ids, rebinned_wfs)
    sipm_r           = SiPMResponses.build_empty_instance()
    expected_peak    = S2(rebinned_times, pmt_r, sipm_r)

    assert len(peaks) == 1
    assert_Peak_equality(peaks[0], expected_peak)


def test_find_peaks_s1_style(pmt_and_sipm_wfs_with_indices):
    times, pmt_wfs, _, pmt_indices, _ = pmt_and_sipm_wfs_with_indices

    pmt_ids      = np.arange(pmt_wfs.shape[0])
    t_slice      = times[pmt_indices]
    time_range   = minmax(      t_slice[0] - 1,      t_slice[-1] + 2)
    length_range = minmax(pmt_indices.size - 1, pmt_indices.size + 2)
    stride       = 1
    rebin_stride = 1

    peaks = pf.find_peaks(pmt_wfs, pmt_indices,
                          time_range, length_range,
                          stride, rebin_stride,
                          S1, pmt_ids)

    pmt_r         =  PMTResponses(pmt_ids, pmt_wfs[:, pmt_indices])
    sipm_r        = SiPMResponses.build_empty_instance()
    expected_peak = S2(times[pmt_indices], pmt_r, sipm_r)

    assert len(peaks) == 1
    assert_Peak_equality(peaks[0], expected_peak)


def test_find_peaks_s2_style(pmt_and_sipm_wfs_with_indices):
    (times, pmt_wfs, sipm_wfs,
     pmt_indices, sipm_indices) = pmt_and_sipm_wfs_with_indices

    pmt_ids      = np.arange(pmt_wfs.shape[0])
    sipm_ids     = np.arange(sipm_wfs.shape[0])
    t_slice      = times[pmt_indices]
    time_range   = minmax(      t_slice[0] - 1,      t_slice[-1] + 2)
    length_range = minmax(pmt_indices.size - 1, pmt_indices.size + 2)
    stride       =  1
    rebin_stride = 40

    peaks = pf.find_peaks(pmt_wfs, pmt_indices,
                          time_range, length_range,
                          stride, rebin_stride,
                          S2, pmt_ids,
                          sipm_wfs    = sipm_wfs,
                          thr_sipm_s2 = -1)

    (rebinned_times,
     rebinned_wfs  ) = pf.rebin_times_and_waveforms(times, pmt_wfs,
                                                    rebin_stride)

    pmt_r            =  PMTResponses( pmt_ids, rebinned_wfs[:, sipm_indices])
    sipm_r           = SiPMResponses(sipm_ids, sipm_wfs    [:, sipm_indices])
    expected_peak    = S2(rebinned_times[sipm_indices], pmt_r, sipm_r)

    assert len(peaks) == 1
    assert_Peak_equality(peaks[0], expected_peak)


def test_get_pmap(s1_and_s2_with_indices):
    (times, pmt_wfs, sipm_wfs,
     s1_indx, s2_indx, sipm_indices,
     s1_params, s2_params) = s1_and_s2_with_indices
    pmt_ids  = np.arange( pmt_wfs.shape[0])
    sipm_ids = np.arange(sipm_wfs.shape[0])


    pmap = pf.get_pmap(pmt_wfs, s1_indx, s2_indx, sipm_wfs,
                       s1_params, s2_params,
                       thr_sipm_s2 = -1,
                       pmt_ids     = pmt_ids)

    (rebinned_times,
     rebinned_wfs  ) = pf.rebin_times_and_waveforms(times,
                                                    pmt_wfs,
                                                    s2_params["rebin_stride"])


    s1 = S1(times[s1_indx],
            PMTResponses ( pmt_ids, pmt_wfs[:, s1_indx]),
            SiPMResponses.build_empty_instance())

    s2 = S2(rebinned_times[sipm_indices],
            PMTResponses ( pmt_ids, rebinned_wfs[:, sipm_indices]),
            SiPMResponses(sipm_ids,     sipm_wfs[:, sipm_indices]))

    expected_pmap = PMap([s1], [s2])
    assert_PMap_equality(pmap, expected_pmap)


@given(times_and_waveforms(), integers(2, 10))
def test_rebin_times_and_waveforms_sum_axis_1_does_not_change(t_and_wf, stride):
    times, wfs = t_and_wf
    _, rb_wfs  = pf.rebin_times_and_waveforms(times, wfs, stride)
    assert np.sum(wfs, axis=1) == approx(np.sum(rb_wfs, axis=1))


@given(times_and_waveforms(), integers(2, 10))
def test_rebin_times_and_waveforms_sum_axis_0_does_not_change(t_and_wf, stride):
    times, wfs = t_and_wf
    sum_wf     = np.stack([np.sum(wfs, axis=0)])
    _, rb_wfs  = pf.rebin_times_and_waveforms(times,     wfs, stride)
    _, rb_sum  = pf.rebin_times_and_waveforms(times, sum_wf , stride)
    assert rb_sum[0] == approx(np.sum(rb_wfs, axis=0))


@given(times_and_waveforms(), integers(2, 10))
def test_rebin_times_and_waveforms_number_of_wfs_does_not_change(t_and_wf, stride):
    times, wfs  = t_and_wf
    _, rb_wfs = pf.rebin_times_and_waveforms(times, wfs, stride)
    assert len(wfs) == len(rb_wfs)


@given(times_and_waveforms(), integers(2, 10))
def test_rebin_times_and_waveforms_number_of_bins_is_correct(t_and_wf, stride):
    times, wfs       = t_and_wf
    rb_times, rb_wfs = pf.rebin_times_and_waveforms(times, wfs, stride)
    expected_n_bins  = times.size // stride
    if times.size % stride != 0:
        expected_n_bins += 1

    assert rb_times.size     == expected_n_bins
    assert rb_wfs  .shape[1] == expected_n_bins


@given(times_and_waveforms())
def test_rebin_times_and_waveforms_stride_1_does_not_rebin(t_and_wf):
    times, wfs       = t_and_wf
    rb_times, rb_wfs = pf.rebin_times_and_waveforms(times, wfs, 1)

    assert np.all(times == rb_times)
    assert wfs == approx(rb_wfs)


@given(times_and_waveforms(), integers(2, 10))
def test_rebin_times_and_waveforms_times_are_consistent(t_and_wf, stride):
    times, wfs  = t_and_wf

    # The samples falling in the last bin cannot be so easily
    # compared as the other ones so I remove them.
    remain = times.size - times.size % stride
    times  = times[:remain]
    wfs    = wfs  [:remain]
    rb_times, _ = pf.rebin_times_and_waveforms(times, np.ones_like(wfs), stride)

    assert np.sum(rb_times) * stride == approx(np.sum(times))
