import numpy as np

from numpy.testing import assert_allclose
from pytest        import          approx
from pytest        import          raises
from pytest        import            mark
from pytest        import         fixture

from hypothesis                import       assume
from hypothesis                import        given
from hypothesis.strategies     import     integers
from hypothesis.strategies     import       floats
from hypothesis.strategies     import sampled_from
from hypothesis.strategies     import    composite
from hypothesis.extra.numpy    import       arrays

from .. core.core_functions  import           weighted_mean_and_std
from .. core.random_sampling import                    NoiseSampler
from .. core                 import                 system_of_units as units
from .. core.testing_utils   import                         exactly
from .. core.testing_utils   import assert_SensorResponses_equality
from .. core.testing_utils   import            assert_Peak_equality
from .. core.testing_utils   import                  previous_float

from invisible_cities.database import load_db as DB

from .  pmaps import  PMTResponses
from .  pmaps import SiPMResponses
from .  pmaps import            S1
from .  pmaps import            S2
from .  pmaps import          PMap
from .  pmaps import    SiPMCharge


wf_min =   0
wf_max = 100


@composite
def sensor_responses(draw, n_samples=None, subtype=None, ids=None):
    n_sensors   = draw(integers(1,  5)) if       ids is None else len(ids)
    n_samples   = draw(integers(1, 50)) if n_samples is None else n_samples
    shape       = n_sensors, n_samples
    all_wfs     = draw(arrays(float,     shape, floats  (wf_min, wf_max)))
    if     ids is None:
        ids     = draw(arrays(  int, n_sensors, integers(0, 1e3), unique=True))
    if subtype is None:
        subtype = draw(sampled_from((PMTResponses, SiPMResponses)))
    args        = np.sort(ids), all_wfs
    return args, subtype(*args)


@composite
def peaks(draw, subtype=None, pmt_ids=None, with_sipms=True):
    nsamples      = draw(integers(1, 20))
    _, pmt_r      = draw(sensor_responses(nsamples,  PMTResponses, pmt_ids))
    sipm_r        = SiPMResponses.build_empty_instance()
    assume(pmt_r.sum_over_sensors[ 0] != 0)
    assume(pmt_r.sum_over_sensors[-1] != 0)

    if subtype is None:
        subtype   = draw(sampled_from((S1, S2)))
    if with_sipms:
        _, sipm_r = draw(sensor_responses(nsamples, SiPMResponses))

    times      = draw(arrays(float, nsamples,
                             floats(min_value=0, max_value=1e3),
                             unique = True).map(sorted))

    bin_widths = np.array([1])
    if len(times) > 1:
        time_differences = np.diff(times)
        bin_widths = np.append(time_differences, max(time_differences))

    args       = times, bin_widths, pmt_r, sipm_r
    return args, subtype(*args)


@composite
def pmaps(draw, pmt_ids=None):
    n_s1 = draw(integers(0, 3))
    n_s2 = draw(integers(0, 3))
    assume(n_s1 + n_s2 > 0)

    s1s  = tuple(draw(peaks(S1, pmt_ids, False))[1] for i in range(n_s1))
    s2s  = tuple(draw(peaks(S2, pmt_ids, True ))[1] for i in range(n_s2))
    args = s1s, s2s
    return args, PMap(*args)


@given(sensor_responses())
def test_SensorResponses_all_waveforms(srs):
    (_, all_waveforms), sr = srs
    assert all_waveforms == approx(sr.all_waveforms)


@given(sensor_responses())
def test_SensorResponses_ids(srs):
    (ids, _), sr = srs
    assert ids == exactly(sr.ids)


@given(sensor_responses())
def test_SensorResponses_waveform(srs):
    (ids, all_waveforms), sr = srs
    for sensor_id, waveform in zip(ids, all_waveforms):
        assert waveform == approx(sr.waveform(sensor_id))


@given(sensor_responses())
def test_SensorResponses_time_slice(srs):
    (_, all_waveforms), sr = srs
    for i, time_slice in enumerate(all_waveforms.T):
        assert time_slice == approx(sr.time_slice(i))


@given(sensor_responses())
def test_SensorResponses_sum_over_times(srs):
    (_, all_waveforms), sr = srs
    assert np.sum(all_waveforms, axis=1) == approx(sr.sum_over_times)


@given(sensor_responses())
def test_SensorResponses_sum_over_sensors(srs):
    (_, all_waveforms), sr = srs
    assert np.sum(all_waveforms, axis=0) == approx(sr.sum_over_sensors)


@mark.parametrize("SR", (PMTResponses, SiPMResponses))
@given(size=integers(1, 10))
def test_SensorResponses_raises_exception_when_shapes_dont_match(SR, size):
    with raises(ValueError):
        sr = SR(np.empty(size),
                np.empty((size + 1, 1)))


@given(peaks())
def test_Peak_sipms(pks):
    (_, _, _, sipm_r), peak = pks
    assert_SensorResponses_equality(sipm_r, peak.sipms)


@given(peaks())
def test_Peak_pmts(pks):
    (_, _, pmt_r, _), peak = pks
    assert_SensorResponses_equality(pmt_r, peak.pmts)


@given(peaks())
def test_Peak_times(pks):
    (times, _, _, _), peak = pks
    assert times == approx(peak.times)


@given(peaks())
def test_Peak_time_at_max_energy(pks):
    _, peak = pks
    index_at_max_energy = np.argmax(peak.pmts.sum_over_sensors)
    assert peak.time_at_max_energy == peak.times[index_at_max_energy]


@given(peaks())
def test_Peak_total_energy(pks):
    _, peak = pks
    assert peak.total_energy == approx(peak.pmts.all_waveforms.sum())


@given(peaks())
def test_Peak_total_charge(pks):
    _, peak = pks
    assert peak.total_charge == approx(peak.sipms.all_waveforms.sum())


@given(peaks())
def test_Peak_height(pks):
    _, peak = pks
    assert peak.height == approx(peak.pmts.sum_over_sensors.max())


#@given(peaks())
#def test_Peak_width(pks):
#    _, peak = pks
#    assert peak.width == approx(peak.times[-1] - peak.times[0])

def test_Peak_width_correct():
    nsamples = 3
    times  = np.arange(nsamples)
    widths = np.full(nsamples, 1)
    pmts   = PMTResponses(np.arange(12), np.full((12, nsamples), 1))

    peak = S1(times, widths, pmts, SiPMResponses.build_empty_instance())
    assert peak.width == nsamples


def _get_indices_above_thr(sr, thr):
    return np.where(sr.sum_over_sensors > thr)[0]


@given(peaks())
def test_Peak_energy_above_threshold_less_than_wf_min(pks):
    _, peak = pks
    sum_wf_min = previous_float(peak.pmts.sum_over_sensors.min())
    assert peak.energy_above_threshold(sum_wf_min) == approx(peak.total_energy)


@given(peaks())
def test_Peak_energy_above_threshold_greater_than_equal_to_wf_max(pks):
    _, peak = pks
    assert peak.energy_above_threshold(peak.height) == 0


@given(peaks(), floats(wf_min, wf_max))
def test_Peak_energy_above_threshold(pks, thr):
    _, peak = pks
    i_above_thr = _get_indices_above_thr(peak.pmts, thr)
    assert (peak.pmts.sum_over_sensors[i_above_thr].sum() ==
            approx(peak.energy_above_threshold(thr)))


@given(peaks())
def test_Peak_charge_above_threshold_less_than_wf_min(pks):
    _, peak = pks
    sum_wf_min = previous_float(peak.sipms.sum_over_sensors.min())
    assert peak.charge_above_threshold(sum_wf_min) == approx(peak.total_charge)


@given(peaks())
def test_Peak_charge_above_threshold_greater_than_equal_to_wf_max(pks):
    _, peak = pks
    sipms_max = peak.sipms.sum_over_sensors.max()
    assert peak.charge_above_threshold(sipms_max) == 0


@given(peaks(), floats(wf_min, wf_max))
def test_Peak_charge_above_threshold(pks, thr):
    _, peak = pks
    i_above_thr = _get_indices_above_thr(peak.sipms, thr)
    assert (peak.sipms.sum_over_sensors[i_above_thr].sum() ==
            approx(peak.charge_above_threshold(thr)))


@given(peaks())
#def test_Peak_width_above_threshold_less_than_wf_min(pks):
def test_Peak_width_above_threshold_with_less_than_wf_min(pks):
    _, peak = pks
    sum_wf_min = previous_float(peak.pmts.sum_over_sensors.min())
    full_width = peak.width_above_threshold(sum_wf_min)
    assert full_width == approx(np.sum(peak.bin_widths))


@given(peaks())
#def test_Peak_width_above_threshold_greater_than_equal_to_wf_max(pks):
def test_Peak_width_above_threshold_max_zero(pks):
    _, peak = pks
    assert peak.width_above_threshold(peak.height) == 0


@given(peaks(), floats(wf_min, wf_max))
def test_Peak_width_above_threshold(pks, thr):
    _, peak = pks
    i_above_thr     = _get_indices_above_thr(peak.pmts, thr)
    expected        = (np.sum(peak.bin_widths[i_above_thr])
                       if np.size(i_above_thr) > 0
                       else 0)
    assert peak.width_above_threshold(thr) == approx(expected)


@given(peaks(), floats(wf_min, wf_max))
def test_Peak_rms_above_threshold(pks, thr):
    _, peak = pks
    i_above_thr     = _get_indices_above_thr(peak.pmts, thr)
    times_above_thr = peak.times[i_above_thr]
    wf_above_thr    = peak.pmts.sum_over_sensors[i_above_thr]
    expected        = (weighted_mean_and_std(times_above_thr, wf_above_thr)[1]
                       if np.size(i_above_thr) > 1 and np.sum(wf_above_thr) > 0
                       else 0)
    assert peak.rms_above_threshold(thr) == approx(expected)


@given(peaks())
def test_Peak_rms_above_threshold_less_than_wf_min(pks):
    _, peak = pks
    sum_wf_min = previous_float(peak.pmts.sum_over_sensors.min())
    assert peak.rms == approx(peak.rms_above_threshold(sum_wf_min))


@given(peaks())
def test_Peak_rms_above_threshold_greater_than_equal_to_wf_max(pks):
    _, peak = pks
    assert peak.rms_above_threshold(peak.height) == 0


@mark.parametrize("PK", (S1, S2))
@given(sr1=sensor_responses(), sr2=sensor_responses())
def test_Peak_raises_exception_when_shapes_dont_match(PK, sr1, sr2):
    with raises(ValueError):
        (ids, wfs), sr1 = sr1
        _         , sr2 = sr2
        n_samples       = wfs.shape[1]
        pk = PK(np.empty(n_samples + 1),
                np.empty(n_samples + 1), sr1, sr2)


@given(pmaps())
def test_PMap_s1s(pmps):
    (s1s, _), pmp = pmps
    assert len(pmp.s1s) == len(s1s)
    for kept_s1, true_s1 in zip(pmp.s1s, s1s):
        assert_Peak_equality(kept_s1, true_s1)


@given(pmaps())
def test_PMap_s2s(pmps):
    (_, s2s), pmp = pmps
    assert len(pmp.s2s) == len(s2s)
    for kept_s2, true_s2 in zip(pmp.s2s, s2s):
        assert_Peak_equality(kept_s2, true_s2)



@fixture(scope='module')
def signal_to_noise_6400():
    return NoiseSampler('new', 6400).signal_to_noise


@fixture(scope='module')
def s2_peak():
    times      = np.arange(20) * units.mus
    bin_widths = np.full_like(times, units.mus)

    pmt_ids  = DB.DataPMT ('new', 6400).SensorID.values
    sipm_ids = DB.DataSiPM('new', 6400).index   .values

    np.random.seed(22)
    pmts  = PMTResponses(pmt_ids,
                         np.random.uniform(0, 100,
                                           (len(pmt_ids),
                                            len(times))))

    sipms = SiPMResponses(sipm_ids,
                          np.random.uniform(0, 10,
                                            (len(sipm_ids),
                                             len(times))))

    ## Values for comparison in tests
    chan_slice      = slice(3, 9)
    raw_arr         = [[7.51759755, 5.60509619, 6.42950506,
                        9.20429423, 9.70312128, 2.02362046],
                       [7.05972334, 2.22846472, 5.46749176,
                        7.31111882, 3.40960906, 4.56294121],
                       [6.70472842, 3.59272265, 5.16309049,
                        1.56536978, 9.81538459, 6.82850322],
                       [4.15689334, 3.41990664, 5.79051761,
                        0.81210519, 1.0304272 , 0.40297561],
                       [4.73786661, 0.6207543 , 1.6485871 ,
                        2.52693314, 8.91532864, 1.52840296],
                       [5.92994762, 3.13080909, 2.56228467,
                        3.18354286, 1.55118873, 0.60703854],
                       [3.51052921, 6.40861592, 4.52017217,
                        1.08859183, 5.93003437, 7.38501235],
                       [8.57725314, 5.32329684, 4.3160815 ,
                        6.77629621, 7.49288644, 9.25124645],
                       [8.0475847 , 6.4163836 , 3.62396899,
                        5.04696816, 8.86944591, 0.08356193],
                       [3.0936211 , 3.95499815, 5.96360799,
                        0.6605126 , 3.86466816, 9.6719478 ],
                       [1.96541611, 5.36489548, 3.70399066,
                        1.12084339, 7.96734339, 3.24741959],
                       [6.14385189, 1.83113282, 2.0741162 ,
                        4.83746891, 3.2362641 , 0.17227037],
                       [1.67604863, 5.59126381, 2.09007632,
                        5.46945561, 9.71342408, 1.6268169 ],
                       [9.61221953, 5.27643349, 2.7304223 ,
                        7.30143289, 2.37011787, 7.83264795],
                       [9.90983548, 5.79429284, 3.23532663,
                        9.17294685, 5.97553658, 4.4452637 ],
                       [0.35683435, 4.30303963, 1.90266857,
                        4.17023551, 4.06186979, 2.7965879 ],
                       [0.83376644, 9.7300586 , 6.05654181,
                        5.09992868, 8.97882656, 4.6280693 ],
                       [8.48222031, 0.12398542, 7.06665329,
                        7.90802805, 8.97576213, 1.98215824],
                       [0.45055814, 9.90187098, 0.7664873 ,
                        4.74533329, 8.9772245 , 6.48808184],
                       [0.88080893, 5.19986295, 6.27365681,
                        4.46247949, 4.42179112, 0.3344096 ]]
    sn_arr          = [[2.55346379, 2.1727307 , 2.35093766,
                        2.86042287, 2.97285019, 1.15426079],
                       [2.46390195, 1.23075187, 2.14164701,
                        2.51344824, 1.6329798 , 1.92597048],
                       [2.3922998 , 1.6673905 , 2.07136149,
                        0.94995374, 2.99152438, 2.43278461],
                       [1.80446599, 1.61757723, 2.21397598,
                        0.57976386, 0.73221374, 0.33431018],
                       [1.95255757, 0.48024565, 1.00356077,
                        1.31780419, 2.83845215, 0.95183676],
                       [2.22861903, 1.5311806 , 1.34814435,
                        1.52928755, 0.98082474, 0.47157412],
                       [1.62612892, 2.34657869, 1.91522882,
                        0.72757474, 2.26079876, 2.54277753],
                       [2.75012934, 2.10862141, 1.86320656,
                        2.40689549, 2.57871696, 2.88240738],
                       [2.65355845, 2.34820047, 1.67626069,
                        2.02740696, 2.83043378, 0.07848485],
                       [1.50175408, 1.76784557, 2.25184422,
                        0.49088564, 1.76128254, 2.95377289],
                       [1.11351605, 2.11819761, 1.69879401,
                        0.74379173, 2.66806122, 1.56651489],
                       [2.27489977, 1.07879322, 1.17346325,
                        1.97695394, 1.58172378, 0.15582134],
                       [0.99700377, 2.16962471, 1.17948397,
                        2.1258517 , 2.97456874, 0.99432725],
                       [2.93003466, 2.09778509, 1.40429925,
                        2.51155634, 1.30074808, 2.62807274],
                       [2.97983427, 2.21480833, 1.56288192,
                        2.85500604, 2.27064999, 1.89628347],
                       [0.29069255, 1.859838  , 1.10727576,
                        1.80798637, 1.81437105, 1.42590517],
                       [0.59196924, 2.96349975, 2.27193322,
                        2.03998311, 2.84951285, 1.94222486],
                       [2.73304224, 0.11442983, 2.48042035,
                        2.62755231, 2.84898002, 1.13831482],
                       [0.35613537, 2.99207933, 0.56870727,
                        1.95439588, 2.8492343 , 2.36312065],
                       [0.61808354, 2.07996798, 2.3182313 ,
                        1.8836441 , 1.90782557, 0.28421464]]
    expected_array  = {SiPMCharge.raw            : raw_arr,
                       SiPMCharge.signal_to_noise:  sn_arr}
    expected_single = {SiPMCharge.raw            : [ 99.6473, 93.8179,
                                                     81.3852, 92.4639,
                                                    125.2603, 75.8990],
                       SiPMCharge.signal_to_noise: [  9.6651,  9.3927,
                                                      8.7087,  9.3396,
                                                     10.9941,  8.4203]}

    return (S2(times, bin_widths, pmts, sipms), chan_slice,
            expected_single, expected_array)


@mark.parametrize("charge_type", SiPMCharge)
def test_sipm_charge_array(charge_type         ,
                           s2_peak             ,
                           signal_to_noise_6400):
    s2_peak, chan_slice, _, expected_array = s2_peak
    charge_tpl = s2_peak.sipm_charge_array(signal_to_noise_6400,
                                           charge_type         ,
                                           single_point = False)

    all_wf     = s2_peak.sipms.all_waveforms
    orig_zeros = np.count_nonzero(all_wf)
    charge_arr = np.array(charge_tpl)
    calc_zeros = np.count_nonzero(charge_arr)
    assert charge_arr.shape == all_wf.T.shape
    assert calc_zeros       == orig_zeros

    calc_slice = np.array(charge_tpl)[:, chan_slice]
    assert_allclose(calc_slice, expected_array[charge_type])


@mark.parametrize("charge_type", SiPMCharge)
def test_sipm_charge_array_single(charge_type         ,
                                  s2_peak             ,
                                  signal_to_noise_6400):
    s2_peak, chan_slice, expected_single, _ = s2_peak
    charge_tpl = s2_peak.sipm_charge_array(signal_to_noise_6400,
                                           charge_type         ,
                                           single_point =  True)

    assert charge_tpl.shape == s2_peak.sipms.ids.shape
    orig_zeros = np.count_nonzero(s2_peak.sipms.sum_over_times)
    calc_zeros = np.count_nonzero(charge_tpl)
    assert calc_zeros == orig_zeros

    assert_allclose(charge_tpl[chan_slice], expected_single[charge_type],
                    atol=5e-5)
