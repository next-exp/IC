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

from . random_sampling  import normalize_distribution
from . random_sampling  import sample_discrete_distribution
from . random_sampling  import uniform_smearing
from . random_sampling  import inverse_cdf_index
from . random_sampling  import inverse_cdf
from . random_sampling  import NoiseSampler
from ..database.load_db import DataSiPM

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
@flaky(max_runs=5, min_passes=2)
def test_uniform_smearing_stats(max_deviation, nsamples):
    expected_mean  = 0
    expected_std   = max_deviation / 3**0.5
    tolerance_mean = 3    * expected_std / nsamples**0.5 # mean within 3 rms
    tolerance_std  = 0.50                                # std  within 50%
    samples        = uniform_smearing(max_deviation, nsamples)
    sample_mean    = np.mean(samples)
    sample_std     = np.std (samples)
    assert np.isclose(sample_mean, expected_mean, atol=tolerance_mean)
    assert np.isclose(sample_std , expected_std , rtol=tolerance_std )


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
def datasipm_run0():
    return DataSiPM(0)


@fixture(scope="module", params=[False, True])
def noise_sampler_run0(request):
    nsamples = 1000
    smear    = request.param
    return NoiseSampler(0, nsamples, smear), nsamples, smear


def test_noise_sampler_output_shape(datasipm_run0, noise_sampler_run0):
    nsipm                        = len(datasipm_run0)
    noise_sampler, nsamples, _   = noise_sampler_run0
    sample                       = noise_sampler.Sample()
    assert sample.shape == (nsipm, nsamples)


def test_noise_sampler_masked_sensors(datasipm_run0, noise_sampler_run0):
    noise_sampler, *_ = noise_sampler_run0
    sample            = noise_sampler.Sample()

    masked_sensors = datasipm_run0[datasipm_run0.Active==0].index.values
    assert not np.any(sample[masked_sensors])
