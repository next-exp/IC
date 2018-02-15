import numpy        as np
import scipy.signal as signal

from pytest        import fixture
from pytest        import approx
from pytest        import mark
from pytest        import raises
from flaky         import flaky

from ..core import core_functions          as cf
from .      import calib_sensors_functions as csf


@fixture
def toy_sipm_signal():
    NSIPM, WL, n_MAU = 100, 100, 100

    common_threshold      = np.random.uniform(0.1, 0.5)
    individual_thresholds = np.random.uniform(0.1, 0.5, size=NSIPM)

    adc_to_pes  = np.full(NSIPM, 100, dtype=np.double)
    signal_adc  = np.random.randint(0, 100, size=(NSIPM, WL), dtype=np.int16)

    # subtract baseline and convert to pes
    signal_pes  = signal_adc - np.mean(signal_adc, axis=1)[:, np.newaxis]
    signal_pes /= adc_to_pes[:, np.newaxis]

    MAU         = np.full(n_MAU, 1 / n_MAU)
    mau         = signal.lfilter(MAU, 1, signal_pes, axis=1)

    over_common_thr     = signal_pes > mau + common_threshold
    over_individual_thr = signal_pes > mau + individual_thresholds[:, np.newaxis]

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
    baseline   = 1000
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

    baseline = 1000
    nsipm    = sipm.shape[0]
    sipm += baseline + np.arange(nsipm).reshape(nsipm, 1) * 10
    return sipm, adc_to_pes


@fixture
def oscillating_waveform_wo_baseline():
    n_samples   = 50000
    noise_sigma = 0.01
    times       = np.linspace(0, 10, n_samples)
    wf          = np.sin(times) + np.random.normal(0, noise_sigma, n_samples)
    wfs         = wf[np.newaxis]
    adc_to_pes  = np.random.uniform(10, 20, size=(1,))
    return wfs, adc_to_pes, n_samples, noise_sigma


@fixture
def oscillating_waveform_with_baseline(oscillating_waveform_wo_baseline):
    wfs, adc_to_pes, n_samples, noise_sigma = oscillating_waveform_wo_baseline
    baseline = np.random.uniform(1000, 2000, size=(1,))
    wfs     += baseline
    return wfs, adc_to_pes, n_samples, noise_sigma, baseline


@flaky(max_runs=2)
@mark.parametrize("kwargs",
                  (dict(bls_mode = csf.BlsMode.mean),
                   dict(bls_mode = csf.BlsMode.mau , n_MAU = 10)))
def test_subtract_baseline_valid_options_sanity_check(gaussian_sipm_signal, kwargs):
    signal, _ = gaussian_sipm_signal
    bls       = csf.subtract_baseline(signal, **kwargs)
    n_1sigma  = np.count_nonzero(cf.in_range(bls, -3, 3))
    assert n_1sigma > 0.99 * signal.size


@mark.parametrize("wrong_bls_mode",
                  (0, "0", 1, "1", None, "None",
                   "mean", "mau", "BlsMode.mean", "BlsMode.mau"))
def test_subtract_baseline_raises_TypeError(gaussian_sipm_signal,
                                             wrong_bls_mode):
    signal, _ = gaussian_sipm_signal
    with raises(TypeError):
        bls = csf.subtract_baseline(signal, bls_mode=wrong_bls_mode)


def test_subtract_baseline_mau_mode_raises_KeyError(gaussian_sipm_signal):
    signal, _ = gaussian_sipm_signal
    with raises(KeyError):
        # This mode needs an extra argument. Assert that it
        # complains if not given.
        bls = csf.subtract_baseline(signal, bls_mode=csf.BlsMode.mau)


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
    n_mau                    = n_samples // 500

    (ccwfs  , ccwfs_mau  ,
     cwf_sum, cwf_sum_mau) = csf.calibrate_pmts(wfs, adc_to_pes,
                                                n_mau, nsigma * noise_sigma)

    # Because there is only one waveform, the sum and the
    # waveform itself must be the same.
    assert ccwfs    .size == cwf_sum    .size
    assert ccwfs_mau.size == cwf_sum_mau.size

    assert ccwfs    [0] == approx(cwf_sum)
    assert ccwfs_mau[0] == approx(cwf_sum_mau)

    assert wfs[0] / adc_to_pes[0] == approx(cwf_sum)

    number_of_zeros = np.count_nonzero(cwf_sum_mau == 0)
    assert number_of_zeros > fraction * cwf_sum_mau.size


@mark.parametrize("nsigma fraction".split(),
                  ((1, 0.68),
                   (2, 0.95),
                   (3, 0.97)))
def test_calibrate_sipms_stat(oscillating_waveform_with_baseline,
                              nsigma, fraction):
    (wfs, adc_to_pes,
     n_samples, noise_sigma,
     baseline )              = oscillating_waveform_with_baseline
    n_mau                    = n_samples // 500

    ccwfs = csf.calibrate_sipms(wfs, adc_to_pes, nsigma * noise_sigma, n_mau)

    number_of_zeros = np.count_nonzero(ccwfs == 0)
    assert number_of_zeros > fraction * ccwfs.size


def test_calibrate_sipms_common_threshold(toy_sipm_signal):
    (signal_adc, adc_to_pes,
     signal_zs_common_threshold, _,
     common_threshold, _) = toy_sipm_signal

    zs_wf = csf.calibrate_sipms(signal_adc, adc_to_pes,
                                common_threshold, n_MAU=100)

    for actual, expected in zip(zs_wf, signal_zs_common_threshold):
        assert actual == approx(expected)


def test_calibrate_sipms_individual_thresholds(toy_sipm_signal):
    (signal_adc, adc_to_pes,
     _, signal_zs_individual_thresholds,
     _, individual_thresholds) = toy_sipm_signal

    zs_wf = csf.calibrate_sipms(signal_adc, adc_to_pes,
                                individual_thresholds, n_MAU=100)
    for actual, expected in zip(zs_wf, signal_zs_individual_thresholds):
        assert actual == approx(expected)


def test_wf_baseline_subtracted_is_close_to_zero(gaussian_sipm_signal):
    sipm_wfs, adc_to_pes = gaussian_sipm_signal
    bls_wf = csf.subtract_baseline_and_calibrate(sipm_wfs, adc_to_pes)
    np.testing.assert_allclose(np.mean(bls_wf, axis=1), 0, atol=1e-10)


def test_wf_baseline_subtracted_mau_is_close_to_zero(gaussian_sipm_signal):
    sipm_wfs, adc_to_pes = gaussian_sipm_signal
    bls_wf = csf.subtract_baseline_mau_and_calibrate(sipm_wfs, adc_to_pes, n_MAU=10)
    np.testing.assert_allclose(np.mean(bls_wf, axis=1), 0, atol=0.1)
