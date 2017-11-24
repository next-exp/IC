import numpy as np

from flaky  import flaky
from pytest import mark
from pytest import fixture

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
from . random_sampling  import NoiseSampler

sensible_sizes    =                  integers(min_value =    2,
                                              max_value =   20)
valid_domains     = lambda size: float_arrays(min_value = -100,
                                              max_value = +100,
                                              size      = size).map(sorted)
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
    assert true_value == icdf


@fixture(scope="module")
def run_number():
    return 4714

@fixture(scope="module")
def datasipm(run_number):
    return DataSiPM(run_number)


@fixture(scope="module", params=[False, True])
def noise_sampler(request, run_number):
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
    return (NoiseSampler(run_number, nsamples, smear),
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
        sample = samples[i]
        if active:
            if smear:
                # Find closest energy bin and ensure it is close enough.
                diffs      = noise_sampler.xbins - sample[:, np.newaxis]
                closest    = np.min(np.abs(diffs), axis=1)
                assert np.all(closest <= noise_sampler.dx)
            else:
                assert np.all(np.in1d(sample, noise_sampler.xbins))
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
