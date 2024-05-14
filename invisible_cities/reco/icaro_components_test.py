import pytest

import numpy          as np
import numpy.testing  as npt
import pandas         as pd
import pandas.testing as pdt

from hypothesis             import given
from hypothesis.strategies  import floats

from .. reco                import icaro_components as icarcomp
from ..types.symbols        import KrFitFunction
from .. evm.ic_containers   import FitFunction


@given(floats(min_value = 0,
              max_value = 10),
       floats(min_value = 1000,
              max_value = 1600),
       floats(min_value = 1e3,
              max_value = 1e5),
       floats(min_value = 1e3,
              max_value = 1e6))
def test_expo_seed_output_values(zmin, zmax, elt, e0):
    x = np.array( [ zmin, zmax ] )
    y = e0 * np.exp( - x / elt )
    e0_test, elt_test = icarcomp.expo_seed(x, y)
    npt.assert_allclose(e0_test,    e0, rtol=0.01)
    npt.assert_allclose(elt_test, -elt, rtol=0.01)


@given(floats(min_value=0, max_value=10),
       floats(min_value=10, max_value=20),
       floats(min_value=1, max_value=100),
       floats(min_value=0, max_value=10))
def test_lin_function_output_values(x_min, x_max, a, b):
    x = np.array([x_min, x_max])
    y = a + b * x
    a_test, b_test = icarcomp.lin_seed(x, y)
    npt.assert_allclose(a_test, a, rtol=0.01)
    npt.assert_allclose(b_test, b, rtol=0.01)


@pytest.fixture
def sample_dataframe():

    data = {
        'DT' : [10, 20, 30, 40, 50],
        'S2e': [50, 45, 42, 41, 41]
    }
    return pd.DataFrame(data)

def test_prepare_data_linear(sample_dataframe):

    x_data, y_data = icarcomp.prepare_data(KrFitFunction.linear, sample_dataframe)

    expected_x = sample_dataframe['DT']
    expected_y = sample_dataframe['S2e']

    pdt.assert_series_equal(x_data, expected_x)
    pdt.assert_series_equal(y_data, expected_y)


def test_prepare_data_expo(sample_dataframe):

    x_data, y_data = icarcomp.prepare_data(KrFitFunction.expo, sample_dataframe)

    expected_x = sample_dataframe['DT']
    expected_y = sample_dataframe['S2e']

    pdt.assert_series_equal(x_data, expected_x)
    pdt.assert_series_equal(y_data, expected_y)


def test_prepare_data_log_lin(sample_dataframe):

    x_data, y_data = icarcomp.prepare_data(KrFitFunction.log_lin, sample_dataframe)

    expected_x = sample_dataframe['DT']
    expected_y = -np.log(sample_dataframe['S2e'])

    pdt.assert_series_equal(x_data, expected_x)
    pdt.assert_series_equal(y_data, expected_y)


def test_get_fit_function_lt():

    expected_functions = {
        KrFitFunction.linear: (lambda x, a, b: a + b * x, icarcomp.lin_seed),
        KrFitFunction.expo: (lambda x, const, mean: const * np.exp(-x / mean), icarcomp.expo_seed),
        KrFitFunction.log_lin: (lambda x, a, b: a + b * x, icarcomp.lin_seed)
    }

    for fit_type, (expected_fit_function, expected_seed_function) in expected_functions.items():
        fit_function, seed_function = icarcomp.get_fit_function_lt(fit_type)

        assert fit_function.__name__  == expected_fit_function.__name__
        assert seed_function.__name__ == expected_seed_function.__name__


@pytest.mark.parametrize("fittype", [KrFitFunction.linear, KrFitFunction.expo, KrFitFunction.log_lin])
@given(floats(min_value=1,  max_value=1e5),
       floats(min_value=10, max_value=1e5))
def test_transform_parameters(fittype, a, b):

    errors = 0.2*np.array([a, b])

    if fittype == KrFitFunction.log_lin:

        fit_output = FitFunction(values=[a, b], errors=errors, cov=np.array([[0.04, 0.02], [0.02, 0.04]]),
                                 fn=None, chi2=None, pvalue=None)

        transformed_par, transformed_err, transformed_cov = icarcomp.transform_parameters(fittype, fit_output)

        E0_expected   = np.exp(-a)
        s_E0_expected = np.abs(E0_expected * errors[0])
        lt_expected   = 1 / b
        s_lt_expected = np.abs(lt_expected**2 * errors[1])
        cov_expected  = E0_expected * lt_expected**2 * 0.02


        npt.assert_allclose(transformed_par, [  E0_expected,   lt_expected], rtol=0.01)
        npt.assert_allclose(transformed_err, [s_E0_expected, s_lt_expected], rtol=0.01)
        assert np.isclose(transformed_cov, cov_expected, rtol=0.01)

    else:

        fit_output = FitFunction(values=[a, b], errors=errors, cov=np.array([[0.04, 0.02], [0.02, 0.04]]),
                                 fn=None, chi2=None, pvalue=None)

        transformed_par, transformed_err, transformed_cov = icarcomp.transform_parameters(fittype, fit_output)

        npt.assert_allclose(transformed_par, [a, b], rtol=0.01)
        npt.assert_allclose(transformed_err, errors, rtol=0.01)

        assert np.isclose(transformed_cov, 0.02, rtol=0.01)
