import numpy as np

from pytest                import mark
from flaky                 import flaky
from hypothesis            import given
from hypothesis.strategies import floats
from hypothesis.strategies import integers
from . testing_utils       import all_elements_close


@flaky(max_runs=2)
@mark.parametrize("  mu  sigma  t_rel  t_abs".split(),
                  (( 10,  5e-3,  1e-1,  1e-6),
                   (  0,  5e-4,  1e-1,  1e-2)))
def test_all_elements_close_simple(mu, sigma, t_rel, t_abs):
    x = np.random.normal(mu, sigma, 10)
    assert all_elements_close(x, t_rel=t_rel, t_abs=t_abs)


@flaky(max_runs=2)
@given(floats  (min_value = -100,
                max_value = +100),
       floats  (min_value =    1,
                max_value =  +10))
def test_all_elements_close_par(mu, sigma):
    x = np.random.normal(mu, sigma, 10)
    assert all_elements_close(x, t_rel=5 * sigma, t_abs=5 * sigma)
