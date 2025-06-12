import numpy          as np
import numpy.testing  as npt
import pandas         as pd
import scipy.optimize as so

from pytest                 import raises
from pytest                 import fixture, mark
from hypothesis             import given, settings
from hypothesis.strategies  import floats, integers

from .. reco                import krmap_functions as krf
from ..types.symbols        import KrFitFunction
from .. evm.ic_containers   import FitFunction
from .. core.fit_functions  import expo


@given(floats(min_value = 0,  max_value = 10),
       floats(min_value = 10, max_value = 20),
       floats(min_value = 1,  max_value = 100),
       floats(min_value = 0,  max_value = 10))
def test_lin_seed_output_values(x_min, x_max, a, b):
    x = np.array([x_min, x_max])
    y = a + b * x
    a_test, b_test = krf.lin_seed(x, y)

    assert np.isclose(a_test, a)
    assert np.isclose(b_test, b)


def test_lin_seed_same_x_values():
    x = np.ones(20)
    y = np.ones(20) * 123
    a_test, b_test = krf.lin_seed(x, y)

    assert a_test == 123
    assert b_test ==   0


@given(floats(min_value = 1,    max_value = 10),
       floats(min_value = 1000, max_value = 1600),
       floats(min_value = 1e4,  max_value = 1e5),
       floats(min_value = 1e4,  max_value = 1e5))
def test_expo_seed_output_values(zmin, zmax, elt, e0):
    x = np.array([zmin, zmax])
    y = e0 * np.exp(-x / elt)
    e0_test, elt_test = krf.expo_seed(x, y)

    assert np.isclose( e0_test,  e0, rtol=0.1)
    assert np.isclose(elt_test, elt, rtol=0.1)


@mark.parametrize('y', ([-1, 1, 1, 1, 1], [1, 1, 1, 1, -1]))
def test_expo_seed_negative_y_values(y):
    x = np.ones(5)
    with raises(ValueError, match = 'y data must be > 0'):
        a_test, b_test = krf.expo_seed(x, y)


@fixture
def sample_df():
    data = {'DT' : [10, 20, 30, 40, 50],
            'S2e': [50, 45, 42, 41, 41]}
    return pd.DataFrame(data)


def test_select_fit_variables(sample_df):
    x_linear,  y_linear  = krf.select_fit_variables(KrFitFunction.linear,  sample_df)
    x_expo,    y_expo    = krf.select_fit_variables(KrFitFunction.expo,    sample_df)
    x_log_lin, y_log_lin = krf.select_fit_variables(KrFitFunction.log_lin, sample_df)

    # First return the same for the 3 cases
    assert (x_expo    == x_linear).all()
    assert (x_log_lin == x_linear).all()

    # Second return different for log_lin case
    assert  (y_linear  == y_expo).all()
    assert  (y_log_lin != y_expo).all()


@settings(deadline=None)
@given(floats  (min_value = 1,    max_value = 10),
       floats  (min_value = 1000, max_value = 1600),
       integers(min_value = 10,   max_value = 1e3),
       floats  (min_value = 1e3,  max_value = 1e5),
       floats  (min_value = 5e3,  max_value = 1e5))
def test_get_function_and_seed_lt_with_data(x_min, x_max, steps, e0, lt):
    x = np.linspace(x_min, x_max, steps)
    y = expo(x, e0, lt)

    fit_func_lin,         seed_func_lin = krf.get_function_and_seed_lt(KrFitFunction.linear)
    fit_func_expo,       seed_func_expo = krf.get_function_and_seed_lt(KrFitFunction.expo)
    fit_func_log_lin, seed_func_log_lin = krf.get_function_and_seed_lt(KrFitFunction.log_lin)

    popt_lin, _     = so.curve_fit(fit_func_lin,     x, y, p0=seed_func_lin    (x, y))
    popt_expo, _    = so.curve_fit(fit_func_expo,    x, y, p0=seed_func_expo   (x, y))
    popt_log_lin, _ = so.curve_fit(fit_func_log_lin, x, y, p0=seed_func_log_lin(x, y))

    assert     np.isclose(popt_lin[0], popt_expo[0],    rtol=1e-1) # The interceipt should be close between lin and expo
    assert not np.isclose(popt_lin[1], popt_expo[1],    rtol=1e-1) # The "lifetime" should be different between lin and expo

    assert     np.isclose(popt_lin[0], popt_log_lin[0], rtol=1e-10) # The lin and log_lin are the same (the only difference is their
    assert     np.isclose(popt_lin[1], popt_log_lin[1], rtol=1e-10) # inputs: s2e or -log(s2e)) so both parameters should be the same
                                                                      # for the purpose of testing this function

@given(floats(min_value = 1,  max_value = 1e5),
       floats(min_value = 10, max_value = 1e5))
def test_transform_parameters(a, b):
    errors     = 0.2*np.array([a, b])
    fit_output = FitFunction(values=[a, b], errors=errors, cov=np.array([[0.04, 0.02], [0.02, 0.04]]),
                             fn=None, chi2=None, pvalue=None, infodict=None, mesg=None, ier=None)

    transformed_par, transformed_err, transformed_cov = krf.transform_parameters(fit_output)

    E0_expected   = np.exp(-a)
    s_E0_expected = np.abs(E0_expected * errors[0])
    lt_expected   = 1 / b
    s_lt_expected = np.abs(lt_expected**2 * errors[1])
    cov_expected  = E0_expected * lt_expected**2 * 0.02

    npt.assert_allclose(transformed_par, [  E0_expected,   lt_expected])
    npt.assert_allclose(transformed_err, [s_E0_expected, s_lt_expected])
    assert np.isclose(transformed_cov, cov_expected)
