import numpy        as np

from functools import reduce
from operator  import add

from pytest import fixture
from pytest import approx
from pytest import mark
from pytest import raises
from flaky  import flaky

from .. core.testing_utils import all_elements_close

from .          import calib_sensors_functions as csf
from .. core    import core_functions          as cf
from .. sierpe  import fee                     as FE
from .. sierpe  import waveform_generator      as wfg

from .. types.symbols import BlsMode

@fixture
def toy_sipm_signal():
    NSIPM, WL = 10, 10

    common_threshold      = np.random.uniform(0.1, 0.5)
    individual_thresholds = np.random.uniform(0.1, 0.5, size=NSIPM)

    adc_to_pes  = np.full(NSIPM, 100, dtype=np.double)
    signal_adc  = np.random.randint(0, 100, size=(NSIPM, WL), dtype=np.int16)

    # subtract baseline and convert to pes
    signal_pes = csf.subtract_baseline_and_calibrate(signal_adc, adc_to_pes,
                                                     bls_mode = BlsMode.mode)

    over_common_thr     = signal_pes > common_threshold
    over_individual_thr = signal_pes > individual_thresholds[:, np.newaxis]

    signal_zs_common_threshold      = np.where(over_common_thr    , signal_pes, 0)
    signal_zs_individual_thresholds = np.where(over_individual_thr, signal_pes, 0)

    return (signal_adc, adc_to_pes,
            signal_zs_common_threshold,
            signal_zs_individual_thresholds,
            common_threshold,
            individual_thresholds)


@fixture
def gaussian_sipm_signal_wo_baseline():
    """
    This fixture generates waveforms gaussianly distributed
    around 0, so that the average of the zs waveform is
    very close to zero.
    """
    nsipm      = 40
    wfl        = 100
    sipm       = np.random.normal(0, 1, size=(nsipm, wfl)).astype(np.int16)
    adc_to_pes = np.abs(np.random.normal(10, 0.1, nsipm))

    return sipm, adc_to_pes


@fixture
def gaussian_sipm_signal(gaussian_sipm_signal_wo_baseline):
    """
    This fixture generates waveforms gaussianly distributed
    around the baseline, so that the average of the zs waveform is
    very close to zero.
    """
    sipm, adc_to_pes = gaussian_sipm_signal_wo_baseline

    baseline  = 1000
    nsipm     = sipm.shape[0]
    sipm     += baseline + np.arange(nsipm).reshape(nsipm, 1) * 10
    return sipm, adc_to_pes


@fixture
def oscillating_waveform_wo_baseline():
    n_samples   = 50000
    noise_sigma = 0.01
    times       = np.linspace(0, 10, n_samples)
    wf          = np.sin(times) + np.random.normal(0, noise_sigma, n_samples)
    wfs         = wf[np.newaxis]
    adc_to_pes  = np.random.uniform(10, 20, size=(1,))
    return wfs.astype(np.int16), adc_to_pes, n_samples, noise_sigma


@fixture
def oscillating_waveform_with_baseline(oscillating_waveform_wo_baseline):
    wfs, adc_to_pes, n_samples, noise_sigma = oscillating_waveform_wo_baseline
    baseline  = np.random.randint(1000, 2000, dtype=np.int16)
    wfs      += baseline
    return wfs, adc_to_pes, n_samples, noise_sigma, baseline


@fixture
def square_pmt_and_sipm_waveforms():
    pedestal = 25000
    nsensors =    5
    fee      = FE.FEE(noise_FEEPMB_rms = 1 * FE.NOISE_I  ,
                      noise_DAQ_rms    = 1 * FE.NOISE_DAQ)
    coef     = fee.freq_LHPFd * np.pi
    wfp      = wfg.WfmPar(w_type    = 'square',
                          t_tot     = 5000 * 25,
                          t_pre     =  600 * 25,
                          t_rise_s2 =   20 * 25,
                          t_flat_s2 =  500 * 25,
                          noise     =    1     ,
                          q_s2      =   50     )

    wfms        = wfg.waveform_generator(fee, wfp, nsensors=nsensors, pedestal=pedestal, random_t0=False)
    pmts_fee    = wfms.fee
    pmts_blr    = wfms.blr
    pmts_BLR    = wfg.deconv_pmts(pmts_fee, coef)
    sipms_wfm   = wfms.blr
    sipms_noped = (sipms_wfm - pedestal).astype(np.int16)

    return pedestal, nsensors, pmts_fee, pmts_blr, pmts_BLR, sipms_wfm, sipms_noped


@flaky(max_runs=2)
def test_subtract_baseline_mode_yield_compatible_result_for_gaussian_signal(gaussian_sipm_signal):
    signal, _ = gaussian_sipm_signal
    n_1sigma = []
    for bls_mode in BlsMode:
        bls   = csf.subtract_baseline(signal, bls_mode=bls_mode)
        n_1sigma.append(np.count_nonzero(cf.in_range(bls, -3, 3)))

    assert all_elements_close(n_1sigma, t_rel=1e-2)


@flaky(max_runs=2)
@mark.parametrize("bls_mode", BlsMode)
def test_subtract_baseline_valid_options_sanity_check(gaussian_sipm_signal, bls_mode):
    signal, _ = gaussian_sipm_signal
    bls       = csf.subtract_baseline(signal, bls_mode=bls_mode)
    n_1sigma  = np.count_nonzero(cf.in_range(bls, -3, 3))
    assert n_1sigma > 0.99 * signal.size


@mark.parametrize("wrong_bls_mode",
                  (0, "0", 1, "1", None, "None",
                   "mean", "maw", "BlsMode.mean", "BlsMode.maw"))
def test_subtract_baseline_raises_TypeError(wrong_bls_mode):
    dummy = np.empty((2, 2))
    with raises(TypeError):
        csf.subtract_baseline(dummy, bls_mode=wrong_bls_mode)


def test_calibrate_wfs(gaussian_sipm_signal_wo_baseline):
    signal_pes, adc_to_pes = gaussian_sipm_signal_wo_baseline
    signal_adc = signal_pes * adc_to_pes[:, np.newaxis]
    calibrated = csf.calibrate_wfs(signal_adc, adc_to_pes)
    n_1sigma   = np.count_nonzero(cf.in_range(calibrated, -3, 3))
    assert n_1sigma > 0.99 * signal_pes.size


def test_calibrate_wfs_with_zeros(gaussian_sipm_signal_wo_baseline):
    signal_pes, adc_to_pes = gaussian_sipm_signal_wo_baseline
    signal_adc             = signal_pes * adc_to_pes[:, np.newaxis]

    dead_index             = np.random.choice(signal_pes.shape[0])
    adc_to_pes[dead_index] = 0

    calibrated = csf.calibrate_wfs(signal_adc, adc_to_pes)
    assert calibrated[dead_index] == approx(np.zeros(signal_pes.shape[1]))


@flaky(max_runs=2)
@mark.parametrize("nsigma fraction".split(),
                  ((1, 0.68),
                   (2, 0.95),
                   (3, 0.97)))
def test_calibrate_pmts_stat(oscillating_waveform_wo_baseline,
                             nsigma, fraction):
    (wfs, adc_to_pes,
     n_samples, noise_sigma) = oscillating_waveform_wo_baseline
    n_maw                    = n_samples // 500

    (ccwfs  , ccwfs_maw  ,
     cwf_sum, cwf_sum_maw) = csf.calibrate_pmts(wfs, adc_to_pes,
                                                n_maw, nsigma * noise_sigma)

    # Because there is only one waveform, the sum and the
    # waveform itself must be the same.
    assert ccwfs    .size == cwf_sum    .size
    assert ccwfs_maw.size == cwf_sum_maw.size

    assert ccwfs    [0] == approx(cwf_sum)
    assert ccwfs_maw[0] == approx(cwf_sum_maw)

    assert wfs[0] / adc_to_pes[0] == approx(cwf_sum)

    number_of_zeros = np.count_nonzero(cwf_sum_maw == 0)
    assert number_of_zeros > fraction * cwf_sum_maw.size


@mark.parametrize("nsigma fraction".split(),
                  ((1, 0.68),
                   (2, 0.95),
                   (3, 0.97)))
def test_calibrate_sipms_stat(oscillating_waveform_with_baseline,
                              nsigma, fraction):
    (wfs, adc_to_pes,
     n_samples, noise_sigma,
     baseline )              = oscillating_waveform_with_baseline
    #n_maw                    = n_samples // 500

    ccwfs = csf.calibrate_sipms(wfs, adc_to_pes, nsigma * noise_sigma, bls_mode=BlsMode.mode)

    number_of_zeros = np.count_nonzero(ccwfs == 0)
    assert number_of_zeros > fraction * ccwfs.size


def test_calibrate_sipms_common_threshold(toy_sipm_signal):
    (signal_adc, adc_to_pes,
     signal_zs_common_threshold, _,
     common_threshold, _) = toy_sipm_signal

    zs_wf = csf.calibrate_sipms(signal_adc, adc_to_pes,
                                common_threshold, bls_mode=BlsMode.mode)

    for actual, expected in zip(zs_wf, signal_zs_common_threshold):
        assert actual == approx(expected)


def test_calibrate_sipms_individual_thresholds(toy_sipm_signal):
    (signal_adc, adc_to_pes,
     _, signal_zs_individual_thresholds,
     _, individual_thresholds) = toy_sipm_signal


    zs_wf = csf.calibrate_sipms(signal_adc, adc_to_pes,
                                individual_thresholds,
                                bls_mode=BlsMode.mode)
    for actual, expected in zip(zs_wf, signal_zs_individual_thresholds):
        assert actual == approx(expected)


def test_wf_baseline_subtracted_is_close_to_zero(gaussian_sipm_signal):
    sipm_wfs, adc_to_pes = gaussian_sipm_signal
    bls_wf = csf.subtract_baseline_and_calibrate(sipm_wfs, adc_to_pes)
    np.testing.assert_allclose(np.mean(bls_wf, axis=1), 0, atol=1e-10)


@flaky(max_runs=10, min_passes=10)
def test_subtract_mean_diff_minmax_remains_constant(square_pmt_and_sipm_waveforms):
    _, _, pmts_fee, _, _, _, _ = square_pmt_and_sipm_waveforms
    pmts_bls = csf.subtract_mean(pmts_fee)

    # Baseline subtraction is only an overall shift,
    # the differences between min and max must remain constant
    wf_diff_original = np.max(pmts_fee, axis=1) - np.min(pmts_fee, axis=1)
    wf_diff_bls      = np.max(pmts_bls, axis=1) - np.min(pmts_bls, axis=1)
    assert np.allclose(wf_diff_bls, wf_diff_original)


@flaky(max_runs=10, min_passes=10)
def test_subtract_mean_mean_is_zero(square_pmt_and_sipm_waveforms):
    _, _, pmts_fee, _, _, _, _ = square_pmt_and_sipm_waveforms
    pmts_bls = csf.subtract_mean(pmts_fee)

    # By definition, if we subtract the mean, the mean
    # of the resulting object must be zero
    wf_mean_bls = np.mean(pmts_bls, axis=1)
    assert np.allclose(wf_mean_bls, 0, atol=1e-10)


@flaky(max_runs=10, min_passes=10)
def test_subtract_mean_difference_is_mean(square_pmt_and_sipm_waveforms):
    _, _, pmts_fee, _, _, _, _ = square_pmt_and_sipm_waveforms
    pmts_bls = csf.subtract_mean(pmts_fee)
    means    = np.mean(pmts_fee, axis=1)

    # If the mean is subtracted, the difference between
    # the original wf and the bls one should be exactly the mean
    # at all points
    diffs = pmts_fee - pmts_bls
    for mean, diff in zip(means, diffs):
        assert np.allclose(diff, mean)


@flaky(max_runs=10, min_passes=10)
def test_mean_for_pmts_fee_is_unbiased(square_pmt_and_sipm_waveforms):
    _, _, pmts_fee, _, _, _, _ = square_pmt_and_sipm_waveforms
    pmts_bls = csf.subtract_mean(pmts_fee) # ped subtracted near zero
    sums     = np.sum(pmts_bls, axis=1)    # cancel fluctuations very close to zeo
    assert np.allclose(sums, 0, atol=1e-2)


def test_areas_pmts_are_close(square_pmt_and_sipm_waveforms):
    _, nsensors, _, _, pmts_BLR, _, _ = square_pmt_and_sipm_waveforms
    adc_to_pes           = np.full(nsensors, 100, dtype=float)
    ccwfs, _, cwf_sum, _ = csf.calibrate_pmts(pmts_BLR, adc_to_pes)
    sums                 = np.sum(ccwfs, axis=1)
    assert all_elements_close(sums, t_rel=1e-2)


def test_area_of_sum_equals_sum_of_areas_pmts(square_pmt_and_sipm_waveforms):
    _, nsensors, _, _, pmts_BLR, _, _ = square_pmt_and_sipm_waveforms
    adc_to_pes           = np.full(nsensors, 100, dtype=float)
    ccwfs, _, cwf_sum, _ = csf.calibrate_pmts(pmts_BLR, adc_to_pes)
    stot                 = np.sum(cwf_sum)
    sums                 = np.sum(ccwfs, axis=1)
    stot2                = reduce(add, sums)
    assert stot == approx(stot2)


def test_area_of_sum_equals_sum_of_areas_sipms(square_pmt_and_sipm_waveforms):
    _, nsensors, _, _, _, sipms_wfm, _ = square_pmt_and_sipm_waveforms
    adc_to_pes = np.full(nsensors, 100, dtype=float)
    cwfs       = csf.calibrate_sipms(sipms_wfm, adc_to_pes, thr=10, bls_mode=BlsMode.mode)
    stot       = np.sum(cwfs[0]) * nsensors
    sums       = np.sum(cwfs, axis=1)
    stot2      = reduce(add, sums)
    assert stot == approx(stot2, rel=1e-3)


def test_mean_for_square_waveform_is_biased(square_pmt_and_sipm_waveforms):
    _, _, _, _, _, sipms_wfm, sipms_noped = square_pmt_and_sipm_waveforms
    sipms_mean = csf.subtract_baseline(sipms_wfm, bls_mode=BlsMode.mean)
    diffs      = sipms_noped - sipms_mean
    assert np.mean(diffs) > 2000


def test_median_for_square_waveform_has_small_bias(square_pmt_and_sipm_waveforms):
    _, _, _, _, _, sipms_wfm, sipms_noped = square_pmt_and_sipm_waveforms
    sipms_median = csf.subtract_baseline(sipms_wfm, bls_mode=BlsMode.median)
    diffs        = sipms_noped - sipms_median
    assert np.mean(diffs) < 10


def test_mode_for_square_waveform_has_no_bias(square_pmt_and_sipm_waveforms):
    _, _, _, _, _, sipms_wfm, sipms_noped = square_pmt_and_sipm_waveforms
    sipms_mode = csf.subtract_baseline(sipms_wfm, bls_mode=BlsMode.mode)
    diffs      = sipms_noped - sipms_mode
    assert np.mean(diffs) == approx(0)
