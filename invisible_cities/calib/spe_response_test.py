import warnings

import numpy as np

from pytest import mark
from pytest import raises
from pytest import fixture
from flaky  import flaky
from numpy.testing import assert_allclose

from .. core.stat_functions import poisson_sigma
from .. core.core_functions import shift_to_bin_centers
from .. core.core_functions import in_range
from .. core                import fit_functions        as fitf
from .                      import spe_response         as spe


@mark.parametrize("         bins               expected ".split(),
                  ((np.linspace(-1, 1, 20),    ( 0,  0) ),
                   (np.linspace(-2, 1, 30),    ( 0, 10) ),
                   (np.linspace(-1, 2, 30),    (10,  0) ),
                   (np.linspace(-1, 0, 10),    ( 0, 10) ),
                   (np.linspace( 0, 1, 10),    ( 8,  0) )))
def test_get_padding(bins, expected):
    actual = spe.get_padding(bins)
    assert actual == expected


@mark.parametrize("  min_integral  scale  poisson_mean  expected".split(),
                  ((     1e-3    ,   1  ,       1     ,     6   ),
                   (     1e-3    ,  10  ,       1     ,     7   ),
                   (     1e-3    ,   1  ,       0.1   ,     3   ),
                   (     1e-6    ,   2  ,       2     ,    13   )))
def test_number_of_gaussians(min_integral, scale, poisson_mean, expected):
    n_gaussians, _ = spe.number_of_gaussians(min_integral, scale, poisson_mean)
    assert n_gaussians == expected


@mark.parametrize("  min_integral  scale  poisson_mean                        expected                             ".split(),
                  ((     1e-1    ,   1  ,       1     ,      np.exp(-1  ) / np.array([1, 1  , 2                  ])),
                   (     1e+0    ,  10  ,       1     , 10 * np.exp(-1  ) / np.array([1, 1  , 2                  ])),
                   (     1e-4    ,   1  ,       0.1   ,      np.exp(-0.1) * np.array([1, 0.1, 0.01 / 2, 0.001 / 6])),
                   (     2e-1    ,   2  ,       2     ,  2 * np.exp(-2  ) * np.array([1, 2  , 4    / 2, 8     / 6]))))
def test_number_of_gaussians_integrals(min_integral, scale, poisson_mean, expected):
    _, integrals = spe.number_of_gaussians(min_integral, scale, poisson_mean)
    assert np.allclose(integrals, expected)


def test_suppress_negative_energy_contribution_symmetric():
    xs = np.linspace(-1, 1, 20)
    ys = np.ones_like(xs)

    suppressed = spe.suppress_negative_energy_contribution(xs, ys)
    assert np.all(suppressed[:10] == 0)
    assert np.all(suppressed[10:] == 1)


def test_suppress_negative_energy_contribution_all_positive():
    xs = np.linspace(0, 1, 20)
    ys = np.ones_like(xs)

    suppressed = spe.suppress_negative_energy_contribution(xs, ys)
    assert np.all(suppressed == 1)


def test_suppress_negative_energy_contribution_all_negative():
    xs = np.linspace(-1, 0, 20, endpoint=False)
    ys = np.ones_like(xs)

    suppressed = spe.suppress_negative_energy_contribution(xs, ys)
    assert np.all(suppressed == 0)


@mark.parametrize("n_gaussians", range(1, 8))
def test_poisson_scaled_gaussians_n_gaussians(n_gaussians):
    xs             = np.linspace(-20, 80, 100)
    scale          =  1
    poisson_mean   =  1
    pedestal_mean  =  0
    pedestal_sigma =  5
    gain           = 20
    gain_sigma     = 10

    sum_of_gaussians = spe.poisson_scaled_gaussians(n_gaussians=n_gaussians)
    expected = fitf.gauss(xs, scale / np.exp(poisson_mean), pedestal_mean, pedestal_sigma)
    for i in range(1, n_gaussians):
        norm      = scale / np.exp(poisson_mean) / np.math.factorial(i)
        mean      =  pedestal_mean     + i * gain
        sigma     = (pedestal_sigma**2 + i * gain_sigma**2)**0.5
        gaussian  = fitf.gauss(xs, norm, mean, sigma)
        expected += spe.suppress_negative_energy_contribution(xs, gaussian)

    actual   = sum_of_gaussians(xs,
                                scale, poisson_mean,
                                pedestal_mean, pedestal_sigma,
                                gain, gain_sigma)

    assert np.allclose(actual, expected)


def test_poisson_scaled_gaussians_min_integral():
    xs             = np.linspace(-20, 80, 100)
    scale          =  1
    poisson_mean   =  1
    pedestal_mean  =  0
    pedestal_sigma =  5
    gain           = 20
    gain_sigma     = 10
    min_integral   = 1e-3

    sum_of_gaussians = spe.poisson_scaled_gaussians(min_integral=min_integral)
    expected = fitf.gauss(xs, scale / np.exp(poisson_mean), pedestal_mean, pedestal_sigma)
    for i in range(1, 6):
        norm      = scale / np.exp(poisson_mean) / np.math.factorial(i)
        mean      =  pedestal_mean     + i * gain
        sigma     = (pedestal_sigma**2 + i * gain_sigma**2)**0.5
        gaussian  = fitf.gauss(xs, norm, mean, sigma)
        expected += spe.suppress_negative_energy_contribution(xs, gaussian)

    actual   = sum_of_gaussians(xs,
                                scale, poisson_mean,
                                pedestal_mean, pedestal_sigma,
                                gain, gain_sigma)

    assert np.allclose(actual, expected)


@mark.parametrize("  first  n_gaussians  min_integral".split(),
                  ((   0  ,    None    ,     None    ),
                   (   1  ,    None    ,     None    ),
                   (   0  ,    1234    ,     1234    ),
                   (   1  ,    1234    ,     1234    )))
def test_poisson_scaled_gaussians_raises_ValueError(first, n_gaussians, min_integral):
    with raises(ValueError):
        spe.poisson_scaled_gaussians(first        =        first,
                                     n_gaussians  =  n_gaussians,
                                     min_integral = min_integral)


@fixture(scope="session")
def dark_spectrum_local():
    bins           = np.linspace(-5, 5, 100)
    nsamples       = int(1e6)
    scale          = 3000
    poisson_mean   =    1.2
    pedestal_mean  =    1
    pedestal_sigma =    1
    gain           =   10
    gain_sigma     =    1.5
    min_integral   =   10

    sum_of_gaussians  = spe.poisson_scaled_gaussians(first=1, min_integral=min_integral)
    parameters        = (bins, nsamples,
                         scale, poisson_mean,
                         pedestal_mean, pedestal_sigma,
                         gain, gain_sigma, min_integral)
    xs                = shift_to_bin_centers(bins)
    pedestal          = fitf.gauss(xs, nsamples, pedestal_mean, pedestal_sigma) * np.diff(xs)[0]
    pedestal         *= np.exp(-poisson_mean)
    signal            = sum_of_gaussians(xs,
                                         scale, poisson_mean,
                                         pedestal_mean, pedestal_sigma,
                                         gain, gain_sigma)
    return parameters, pedestal + signal


@fixture(scope="session")
def dark_spectrum_global():
    bins           = np.linspace(-10, 90, 100)
    nsamples       = int(1e6)
    scale          = 3000
    poisson_mean   =    1.2
    pedestal_mean  =    1
    pedestal_sigma =    1
    gain           =   10
    gain_sigma     =    1.5
    min_integral   =   10

    sum_of_gaussians  = spe.poisson_scaled_gaussians(first=1, min_integral=min_integral)
    parameters        = (bins, nsamples,
                         scale, poisson_mean,
                         pedestal_mean, pedestal_sigma,
                         gain, gain_sigma, min_integral)
    xs                = shift_to_bin_centers(bins)
    pedestal          = fitf.gauss(xs, nsamples, pedestal_mean, pedestal_sigma)
    pedestal         *= np.exp(-poisson_mean)
    signal            = sum_of_gaussians(xs,
                                         scale, poisson_mean,
                                         pedestal_mean, pedestal_sigma,
                                         gain, gain_sigma)
    return parameters, pedestal + signal


def test_scaled_dark_pedestal_pedestal(dark_spectrum_local):
    (bins, nsamples, scale, poisson_mean,
     pedestal_mean, pedestal_sigma,
     gain, gain_sigma, min_integral), expected_spectrum = dark_spectrum_local

    xs       = shift_to_bin_centers(bins)
    pedestal = spe.binned_gaussian_spectrum(pedestal_mean, pedestal_sigma, nsamples, bins)
    f        = spe.scaled_dark_pedestal(pedestal,
                                        pedestal_mean, pedestal_sigma,
                                        min_integral)
    actual_spectrum = f(xs, scale, poisson_mean,
                        gain, gain_sigma)

    x0, s0    = pedestal_mean, pedestal_sigma
    selection = in_range(shift_to_bin_centers(bins),   x0 - 5 * s0,    x0 + 5 * s0)
    pull      = expected_spectrum[selection]   -  actual_spectrum[selection]
    pull     /= expected_spectrum[selection]**0.5
    assert np.all(in_range(pull, -2.5, 2.5))


def test_scaled_dark_pedestal_spe(dark_spectrum_global):
    # Test that the spectrum we get is identical ignoring the pedestal
    (bins, nsamples, scale, poisson_mean,
     pedestal_mean, pedestal_sigma,
     gain, gain_sigma, min_integral), expected_spectrum = dark_spectrum_global

    xs       = shift_to_bin_centers(bins)
    pedestal = spe.binned_gaussian_spectrum(pedestal_mean, pedestal_sigma, nsamples, bins)
    f        = spe.scaled_dark_pedestal(pedestal,
                                        pedestal_mean, pedestal_sigma,
                                        min_integral)
    actual_spectrum = f(xs, scale, poisson_mean,
                        gain, gain_sigma)

    x0, s0    = pedestal_mean, pedestal_sigma
    selection = shift_to_bin_centers(bins) > x0 + 10 * s0
    selection = in_range(shift_to_bin_centers(bins),   x0 + 10 * s0,     np.inf)
    assert np.allclose(actual_spectrum[selection], expected_spectrum[selection])


@fixture(scope="session")
def custom_dark_convolution():
    bins           = np.linspace(-20.5, 20.5, 21, endpoint=False)
    nsamples       = int(1e6)
    pedestal_mean  =   0
    pedestal_sigma =   5
    min_integral   = 100
    scale          = 1e4
    poisson_mean   = 1.5
    gain           =  20
    gain_sigma     =  10
    parameters     = ((bins, nsamples, pedestal_mean, pedestal_sigma, min_integral),
                      (scale, poisson_mean, gain, gain_sigma))
    result         = [ 1.65116319e+01, 7.94362759e+01, 2.78701615e+02, 8.76504734e+02, 2.26851240e+03,
                       5.31278424e+03, 1.03944878e+04, 1.74593600e+04, 2.55589484e+04, 3.19861111e+04,
                       3.45738954e+04, 3.20515685e+04, 2.55703000e+04, 1.76763627e+04, 1.04420292e+04,
                       5.33433733e+03, 2.38671060e+03, 9.48032798e+02, 3.60657931e+02, 1.46318948e+02]

    return parameters, np.array(result)


@flaky(max_runs=5, min_passes=5)
def test_dark_convolution(custom_dark_convolution):
    parameters, expected = custom_dark_convolution
    bins, nsamples, pedestal_mean, pedestal_sigma, min_integral  = parameters[0]
    xs        = shift_to_bin_centers(bins)
    pedestal  = spe.binned_gaussian_spectrum(pedestal_mean, pedestal_sigma, nsamples, bins)
    function  = spe.dark_convolution(bins, pedestal, min_integral)
    actual    = function(xs, *parameters[1])
    pull      = expected   -  actual
    pull     /= expected**0.5
    assert np.all(in_range(pull, -2.5, 2.5))


@mark.parametrize("kwargs",
                  (dict(n_gaussians  =   7),
                   dict(min_integral = 100)))
def test_fit_sum_of_gaussians(kwargs):
    fit_function    = spe.poisson_scaled_gaussians(**kwargs)
    true_parameters = np.array([1e4, 1.5, 0, 5, 20, 10])
    seed            = np.random.normal(true_parameters, 1e-2 * true_parameters + 1e-6)

    x = shift_to_bin_centers(np.linspace(-20.5, 200.5, 221, endpoint=False))
    y = fit_function(x, *true_parameters)
    e = poisson_sigma(y, default=1e-3)
    # Suppress warning that can be raised when
    # covariance matrix not calculated properly.
    # Not relevant for test.
    with warnings.catch_warnings():
        from scipy.optimize import OptimizeWarning
        warnings.simplefilter("ignore", category=OptimizeWarning)
        f = fitf.fit(fit_function, x, y, seed, sigma=e)

    assert np.allclose(f.values, true_parameters, rtol=0.05, atol=0.05)


def test_fit_scaled_dark_pedestal():
    bins            = np.linspace(-20.5, 200.5, 221, endpoint=False)
    nsamples        = int(1e6)
    pedestal_mean   =   0
    pedestal_sigma  =   5
    min_integral    = 100
    pedestal        = spe.binned_gaussian_spectrum(pedestal_mean, pedestal_sigma, nsamples, bins)
    fit_function    = spe.scaled_dark_pedestal(pedestal, pedestal_mean, pedestal_sigma, min_integral)
    true_parameters = np.array([1e4, 1.5, 20, 10])
    seed            = np.random.normal(true_parameters, 1e-2 * true_parameters + 1e-6)

    x = shift_to_bin_centers(bins)
    y = fit_function(x, *true_parameters)
    e = poisson_sigma(y, default=1e-3)
    f = fitf.fit(fit_function, x, y, seed, sigma=e)

    assert np.allclose(f.values, true_parameters, rtol=0.05, atol=0.05)


def test_fit_dark_convolution(custom_dark_convolution):
    parameters, _ = custom_dark_convolution
    (bins, nsamples,
     pedestal_mean ,
     pedestal_sigma,
     min_integral  ), true_parameters = parameters

    true_parameters = np.array(true_parameters)
    pedestal        = spe.binned_gaussian_spectrum(pedestal_mean, pedestal_sigma, nsamples, bins)
    fit_function    = spe.dark_convolution(bins, pedestal, min_integral)
    seed            = np.random.normal(true_parameters, 1e-2 * true_parameters + 1e-6)

    x = shift_to_bin_centers(bins)
    y = fit_function (x, *true_parameters)
    e = poisson_sigma(y, default=1e-3)
    f = fitf.fit(fit_function, x, y, seed, sigma=e)

    assert_allclose(f.values, true_parameters, rtol=0.05, atol=0.05)
