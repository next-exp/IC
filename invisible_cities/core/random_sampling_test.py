import numpy as np

from flaky  import flaky
from pytest import mark
from pytest import fixture
from pytest import approx

from hypothesis            import given
from hypothesis.strategies import composite
from hypothesis.strategies import integers
from hypothesis.strategies import floats

from . testing_utils    import float_arrays
from . core_functions   import in_range
from ..database.load_db import DataSiPM

from . random_sampling  import normalize_distribution
from . random_sampling  import sample_discrete_distribution
from . random_sampling  import uniform_smearing
from . random_sampling  import inverse_cdf_index
from . random_sampling  import inverse_cdf
from . random_sampling  import pad_pdfs
from . random_sampling  import DarkModel
from . random_sampling  import NoiseSampler

sensible_sizes    =                  integers(min_value =    2,
                                              max_value =   20)
valid_domains     = lambda size: float_arrays(min_value = -100,
                                              max_value = +100,
                                              size      = size,
                                              unique    = True).map(np.sort)
valid_frequencies = lambda size: float_arrays(min_value =    0,
                                              max_value =  1e4,
                                              size      = size).filter(lambda x: np.sum(x) != 0)
zero_frequencies  = lambda size: float_arrays(min_value =    0,
                                              max_value =    0,
                                              size      = size)

@composite
def valid_distributions(draw):
    size        = draw(sensible_sizes)
    domain      = draw(valid_domains    (size))
    frequencies = draw(valid_frequencies(size))
    return domain, frequencies


@composite
def invalid_distributions(draw):
    size        = draw(sensible_sizes)
    domain      = draw(valid_domains   (size))
    frequencies = draw(zero_frequencies(size))
    return domain, frequencies


cdf_from_freq = lambda x: np.cumsum(x/np.sum(x))

@given(valid_distributions())
def test_normalize_distribution_integral_is_one(distribution):
    _, f         = distribution
    f_normalized = normalize_distribution(f)
    assert np.isclose(f_normalized.sum(), 1)


@given(invalid_distributions())
def test_normalize_distribution_invalid_input(distribution):
    _, f         = distribution
    f_normalized = normalize_distribution(f)
    assert np.array_equal(f_normalized, f)


@given(valid_distributions(),
       integers(min_value = 1, max_value = 10))
def test_sample_discrete_distribution_valid_input(distribution, nsamples):
    domain, frequencies = distribution
    frequencies = normalize_distribution(frequencies)
    samples = sample_discrete_distribution(domain, frequencies, nsamples)
    assert np.all(np.in1d(samples, domain))


@given(invalid_distributions(),
       integers(min_value = 1, max_value = 10))
def test_sample_discrete_distribution_invalid_input(distribution, nsamples):
    domain, frequencies = distribution
    samples = sample_discrete_distribution(domain, frequencies, nsamples)
    assert not np.any(samples)


@given(floats(min_value = 1e-2,
              max_value = 1e+2),
       sensible_sizes)
def test_uniform_smearing_range(max_deviation, nsamples):
    samples = uniform_smearing(max_deviation, nsamples)
    assert np.all(in_range(samples, -max_deviation, max_deviation))


@given(floats  (min_value = 1e-2,
                max_value = 1e+2),
       integers(min_value = 200,
                max_value = 1000))
@flaky(max_runs=10, min_passes=1)
def test_uniform_smearing_stats(max_deviation, nsamples):
    expected_mean  = 0
    expected_std   = max_deviation / 3**0.5
    tolerance_mean = 5 * expected_std / nsamples**0.5 # mean within 5 sigma
    tolerance_std  = 0.20                             # std  within 20%
    samples        = uniform_smearing(max_deviation, nsamples)
    sample_mean    = np.mean(samples)
    sample_std     = np.std (samples)
    assert np.isclose(sample_mean, expected_mean,    atol=tolerance_mean)
    assert np.isclose(sample_std / expected_std , 1, rtol=tolerance_std )


@mark.parametrize("frequencies percentile true_index".split(),
                  ((np.linspace(0, 1,  10), 0.6,  6),
                   (np.linspace(0, 1, 101), 0.6, 60)))
def test_inverse_cdf_index_ad_hoc(frequencies, percentile, true_index):
    icdf_i = inverse_cdf_index(frequencies, percentile)
    assert icdf_i == true_index


@given(valid_distributions(),
       floats(min_value = 0.1, max_value = 0.9))
def test_inverse_cdf_index_hypothesis_generated(distribution, percentile):
    _, freq = distribution
    cdf = cdf_from_freq(freq)
    for i, cp in enumerate(cdf):
        if cp >= percentile:
            true_index = i
            break
    icdf_i = inverse_cdf_index(cdf, percentile)
    assert true_index == icdf_i


@mark.parametrize("bins           expected_length ".split(),
                  ((np.linspace(-1, 1, 20),    20 ),
                   (np.linspace(-2, 1, 30),    40 ),
                   (np.linspace(-1, 2, 30),    40 ),
                   (np.linspace(-1, 0, 10),    20 ),
                   (np.linspace( 0, 1, 10),    18 )))
def test_pad_pdfs(bins, expected_length):
    ## Dummy spectra for 2 'sensors'
    spectra        = np.full((2, len(bins)), 0.1)
    padded_spectra = pad_pdfs(bins, spectra)
    assert padded_spectra.shape[1] == expected_length
    

## @mark.parametrize("bins",
##                   (np.arange(-1, 1, 0.01),
##                    np.arange(-2, 1, 0.01),
##                    np.arange(-1, 2, 0.01)))
## def test_pad_pdfs_xbins(bins):
##     padded_bins = pad_pdfs(bins)

##     bin_diffs = np.diff(padded_bins)
##     assert np.allclose(bin_diffs, bin_diffs[0])
    

@mark.parametrize("domain frequencies percentile true_value".split(),
                  ((np.arange( 10), np.linspace(0, 1,  10), 0.6,  6),
                   (np.arange(101), np.linspace(0, 1, 101), 0.6, 60)))
def test_inverse_cdf_ad_hoc(domain, frequencies, percentile, true_value):
    icdf = inverse_cdf(domain, frequencies, percentile)
    assert true_value == icdf


@given(valid_distributions(),
       floats(min_value = 0.1, max_value = 0.9))
def test_inverse_cdf_hypothesis_generated(distribution, percentile):
    domain, freq = distribution
    cdf = cdf_from_freq(freq)
    for i, (d, cp) in enumerate(zip(domain, cdf)):
        if cp >= percentile:
            true_value = d
            break
    icdf = inverse_cdf(domain, cdf, percentile)
    assert icdf == approx(true_value)


@fixture(scope="module")
def run_number():
    return 4714

@fixture(scope="module")
def datasipm(run_number, dbnew):
    return DataSiPM(dbnew, run_number)


@fixture(scope="module", params=[False, True])
def noise_sampler(request, dbnew, run_number):
    nsamples = 1000
    smear    = request.param
    thr      = 0.99
    true_threshold_counts = {
    0.85  :  19,
    0.95  : 687,
    1.05  : 829,
    1.15  : 199,
    1.25  :  32,
    1.35  :   8,
    1.45  :   6,
    1.55  :   3,
    np.inf:   9}
    return (NoiseSampler(dbnew, run_number, nsamples, smear),
            nsamples, smear,
            thr, true_threshold_counts)


def test_noise_sampler_output_shape(datasipm, noise_sampler):
    nsipm                       = len(datasipm)
    noise_sampler, nsamples, *_ = noise_sampler
    sample                      = noise_sampler.sample()
    assert sample.shape == (nsipm, nsamples)


def test_noise_sampler_masked_sensors(datasipm, noise_sampler):
    noise_sampler, *_ = noise_sampler
    sample            = noise_sampler.sample()

    masked_sensors = datasipm[datasipm.Active==0].index.values
    assert not np.any(sample[masked_sensors])


def test_noise_sampler_attributes(datasipm, noise_sampler):
    true_av_noise     = 0.00333333333333333
    noise_sampler, *_ = noise_sampler

    av_noise = noise_sampler.probs.mean(axis=1)
    active   = datasipm.Active.values
    masked   = active == 0; masked
    assert np.allclose(av_noise[active == 0],             0, rtol = 1e-8)
    assert np.allclose(av_noise[active == 1], true_av_noise, rtol = 1e-8)


def test_noise_sampler_take_sample(datasipm, noise_sampler):
    noise_sampler, _, smear, *_ = noise_sampler
    samples = noise_sampler.sample()
    for i, active in enumerate(datasipm.Active):
        sample        = samples[i]
        adc_to_pe     = noise_sampler.adc_to_pes[i]
        baseline      = noise_sampler.baselines[i]
        bins_adc      = noise_sampler.xbins * adc_to_pe
        bin_width_adc = noise_sampler.dx * adc_to_pe
        if active:
            if smear:
                # Find closest energy bin and ensure it is close enough.
                diffs      = bins_adc + baseline - sample[:, np.newaxis]
                closest    = np.min(np.abs(diffs), axis=1)
                assert np.all(closest <= bin_width_adc)
            else:
                assert np.all(np.in1d(sample, bins_adc + baseline))
        else:
            assert not np.any(sample)


@mark.parametrize("pes_to_adc",
                  (0.25, 1, 2.5, 10))
@mark.parametrize("as_array",
                  (False, True))
def test_noise_sampler_compute_thresholds(datasipm, noise_sampler, pes_to_adc, as_array):
    noise_sampler, *_, thr, true_threshold_counts = noise_sampler

    true_threshold_counts = {pes_to_adc*k: v for k,v in true_threshold_counts.items()}
    if as_array:
        pes_to_adc *= np.ones(len(datasipm))

    thresholds       = noise_sampler.compute_thresholds(thr, pes_to_adc)
    threshold_counts = np.unique(thresholds, return_counts = True)
    threshold_counts = dict(zip(*threshold_counts))

    assert sorted(true_threshold_counts) == sorted(threshold_counts)
    for i, truth in true_threshold_counts.items():
        assert truth == threshold_counts[i]


def test_noise_sampler_multi_sample_distributions(noise_sampler):
    noise_sampler, *_ = noise_sampler

    active       = noise_sampler.active.flatten() != 0
    padded_xbins = pad_pdfs(noise_sampler.xbins)

    nsipm = np.count_nonzero(active)
    PDF_means = np.average(np.repeat(noise_sampler.xbins[None, :], nsipm, 0),
                           axis = 1, weights = noise_sampler.probs[active])

    twomu_pdfs = noise_sampler.multi_sample_distributions(2)

    assert twomu_pdfs.shape[0] == noise_sampler.probs.shape[0]

    twomu_means = np.average(np.repeat(padded_xbins[None, :], nsipm, 0),
                             axis = 1, weights = twomu_pdfs[active])
    assert np.all(twomu_means > PDF_means)


@mark.parametrize("sample_width dark_model".split(),
                  ((2, DarkModel.mean),
                   (3, DarkModel.threshold),
                   (5, DarkModel.threshold)))
def test_noise_sampler_dark_expectation(noise_sampler,
                                        sample_width ,
                                        dark_model   ):
    noise_sampler, *_ = noise_sampler

    dark_mean = noise_sampler.dark_expectation(sample_width, dark_model)

    assert len(dark_mean) == noise_sampler.nsensors
    assert np.any(dark_mean)
    assert np.count_nonzero(dark_mean) == np.count_nonzero(noise_sampler.active)


@mark.parametrize(" ids qs width model expected".split(),
                  (([  0,   1,   2], np.array([ 6, 10,  4]), 2, DarkModel.mean, [2.42, 3.13, 1.96]),
                   ([700, 701, 702], np.array([22,  4, 19]), 4, DarkModel.mean, [4.63, 1.90, 4.30]),
                   ([658, 666, 674], np.array([ 5, 15,  5]), 5, DarkModel.threshold, [1.90, 3.60, 1.88])))
def test_noise_sampler_signal_to_noise(noise_sampler,
                                       ids, qs, width, model, expected):
    noise_sampler, *_ = noise_sampler

    signal_to_noise = noise_sampler.signal_to_noise(ids, qs, width, model)

    assert len(signal_to_noise) == len(ids)
    assert np.allclose(np.round(signal_to_noise, 2), expected)
