import numpy as np

from pytest                       import mark
from flaky                        import flaky
from hypothesis                   import given
from hypothesis.strategies        import floats
from hypothesis.     extra.pandas import data_frames
from hypothesis.     extra.pandas import column
from hypothesis.     extra.pandas import range_indexes
from . testing_utils              import all_elements_close
from . testing_utils              import assert_tables_equality


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

@given(data_frames([column('A', dtype=int  ),
                    column('B', dtype=float),
                    column('C', dtype=str  )],
                   index = range_indexes(max_size=5)))
def test_assert_tables_equality(df):
    table = df.to_records(index=False)
    assert_tables_equality(table, table)

def test_assert_tables_equality_withNaN():
    table = np.array([('Rex', 9, 81.0), ('Fido', 3, np.nan)],
                     dtype=[('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
    assert_tables_equality(table, table)
