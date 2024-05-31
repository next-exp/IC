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


@given(floats(min_value= 0, max_value= 10),
       floats(min_value=10, max_value= 20),
       floats(min_value= 1, max_value=100),
       floats(min_value= 0, max_value= 10))
def test_lin_function_output_values(x_min, x_max, a, b):
    x = np.array([x_min, x_max])
    y = a + b * x
    a_test, b_test = icarcomp.lin_seed(x, y)
    npt.assert_allclose(a_test, a)
    npt.assert_allclose(b_test, b)


@given(floats(min_value = 1,    max_value = 10),
       floats(min_value = 1000, max_value = 1600),
       floats(min_value = 1e2,  max_value = 1e5),
       floats(min_value = 1e3,  max_value = 1e6))
def test_expo_seed_output_values(zmin, zmax, elt, e0):
    x = np.array( [ zmin, zmax ] )
    y = e0 * np.exp( - x / elt )
    e0_test, elt_test = icarcomp.expo_seed(x, y)
    npt.assert_allclose(e0_test,    e0, rtol=0.1)
    npt.assert_allclose(elt_test, -elt, rtol=0.1)


@pytest.fixture
def sample_df():

    data = {
        'DT' : [10, 20, 30, 40, 50],
        'S2e': [50, 45, 42, 41, 41]
    }
    return pd.DataFrame(data)


def test_select_fit_variables(sample_df):

    x_linear,  y_linear  = icarcomp.select_fit_variables(KrFitFunction.linear,  sample_df)
    x_expo,    y_expo    = icarcomp.select_fit_variables(KrFitFunction.expo,    sample_df)
    x_log_lin, y_log_lin = icarcomp.select_fit_variables(KrFitFunction.log_lin, sample_df)

    # First return always the same
    assert (x_linear  == sample_df.DT).all()
    assert (x_expo    == sample_df.DT).all()
    assert (x_log_lin == sample_df.DT).all()

    # Second return different for log_lin case
    assert     (y_linear  ==  sample_df.S2e).all()
    assert     (y_expo    ==  sample_df.S2e).all()
    assert not (y_log_lin ==  sample_df.S2e).all()
    assert     (y_log_lin == -np.log(sample_df.S2e)).all()


def test_get_fit_function_lt():

    # try to fit data with the (in)correct function and check that the result is (un)reasonable

    assert True


@given(floats(min_value=1,  max_value=1e5),
       floats(min_value=10, max_value=1e5))
def test_transform_parameters(a, b):

    errors = 0.2*np.array([a, b])

    fit_output = FitFunction(values=[a, b], errors=errors, cov=np.array([[0.04, 0.02], [0.02, 0.04]]),
                             fn=None, chi2=None, pvalue=None, infodict=None, mesg=None, ier=None)

    transformed_par, transformed_err, transformed_cov = icarcomp.transform_parameters(fit_output)

    E0_expected   = np.exp(-a)
    s_E0_expected = np.abs(E0_expected * errors[0])
    lt_expected   = 1 / b
    s_lt_expected = np.abs(lt_expected**2 * errors[1])
    cov_expected  = E0_expected * lt_expected**2 * 0.02

    npt.assert_allclose(transformed_par, [  E0_expected,   lt_expected])
    npt.assert_allclose(transformed_err, [s_E0_expected, s_lt_expected])
    assert np.isclose(transformed_cov, cov_expected)
