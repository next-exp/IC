import numpy  as np

from pytest                 import approx
from numpy.testing          import assert_array_equal
from numpy.testing          import assert_equal
from numpy.testing          import assert_allclose
from hypothesis.strategies  import integers
from hypothesis.strategies  import floats
from hypothesis.extra.numpy import arrays

from . core_functions import relative_difference


def exactly(value, **kwargs):
    """
    Assert that two numbers (or two sets of numbers) are equal to each
    other with zero tolerance. This function is intended to have a
    similar behavior to `pytest`'s approx function, but for integers.
    This is particularly interesting when comparing numpy arrays, as
    the expression is less readable:

    ```
    np.all(array1 == array2)
    ```

    to be compared with

    ```
    array1 == exactly(array2)
    ```

    which is more pleasant to the human eye and keeps the symmetry with
    floating point comparisons.
    """
    return approx(value, rel=0, abs=0, **kwargs)


def all_elements_close(x, t_rel=1e-3, t_abs=1e-6):
    """Tests that all the elements of a sequence are close within
    relative of abs tolerance

    """
    x  = np.asarray(x)
    x0 = x[0]
    rel_diff = relative_difference(x0, x)
    abs_diff = np.abs(x0 - x)

    return (not np.any(rel_diff > t_rel) or
            not np.any(abs_diff > t_abs))


def previous_float(x):
    """
    Return the next float towards -inf.
    """
    return np.nextafter(x, -np.inf)


def next_float(x):
    """
    Return the next float towards +inf.
    """
    return np.nextafter(x, +np.inf)


def float_arrays(size       =   100,
                 min_value  = -1e20,
                 max_value  = +1e20,
                 mask       =  None,
                 unique     = False,
                 **kwargs          ):
    elements = floats(min_value,
                      max_value,
                      **kwargs)
    if mask is not None:
        elements = elements.filter(mask)
    return arrays(dtype    = np.float64,
                  shape    =       size,
                  elements =   elements,
                  unique   =     unique)


def FLOAT_ARRAY(*args, **kwargs):
    return float_arrays(*args, **kwargs).example()


def random_length_float_arrays(min_length =     0,
                               max_length =   100,
                               **kwargs          ):
    lengths = integers(min_length,
                       max_length)

    return lengths.flatmap(lambda n: float_arrays(       n,
                                                  **kwargs))


def _compare_dataframes(assertion, df1, df2, check_types=True, **kwargs):
    assert sorted(df1.columns) == sorted(df2.columns), "DataFrames with different structure cannot be compared"

    for col in df1.columns:
        col1 = df1[col]
        col2 = df2[col]
        if check_types:
            assert col1.dtype == col2.dtype
        assertion(col1.values, col2.values, **kwargs)


def assert_dataframes_equal(df1, df2, check_types=True, **kwargs):
    _compare_dataframes(assert_array_equal, df1, df2, check_types, **kwargs)


def assert_dataframes_close(df1, df2, check_types=True, **kwargs):
    _compare_dataframes(assert_allclose, df1, df2, check_types, **kwargs)


def assert_SensorResponses_equality(sr0, sr1):
    # This is sufficient to assert equality since all of SensorResponses other
    # properties depend solely on .all_waveforms, and all of those properties
    # are tested.
    assert sr0.ids           == exactly(sr1.ids)
    assert sr0.all_waveforms == approx(sr1.all_waveforms)


def assert_Peak_equality(pk0, pk1):
    assert pk0.times == approx(pk1.times)
    assert_SensorResponses_equality(pk0.pmts , pk1.pmts )
    assert_SensorResponses_equality(pk0.sipms, pk1.sipms)


def assert_PMap_equality(pmp0, pmp1):
    assert len(pmp0.s1s) == len(pmp1.s1s)
    assert len(pmp0.s2s) == len(pmp1.s2s)

    for s1_0, s1_1 in zip(pmp0.s1s, pmp1.s1s):
        assert_Peak_equality(s1_0, s1_1)

    for s2_0, s2_1 in zip(pmp0.s2s, pmp1.s2s):
        assert_Peak_equality(s2_0, s2_1)


def assert_tables_equality(got_table, expected_table):
    table_got      =      got_table[:]
    table_expected = expected_table[:]
    assert len(table_got      ) == len(table_expected      )
    assert len(table_got.dtype) == len(table_expected.dtype)

    if table_got.dtype.names is not None:
        for col_name in table_got.dtype.names:
            got      = table_got     [col_name]
            expected = table_expected[col_name]
            assert type(got) == type(expected)
            try:
                assert_allclose(got, expected)
            except TypeError:
                assert_equal   (got, expected)
    else:
        try:
            assert_allclose(got_table, expected_table)
        except TypeError:
            assert_equal   (got_table, expected_table)

def assert_cluster_equality(a_cluster, b_cluster):
    assert np.allclose(a_cluster.posxy , b_cluster.posxy )
    assert np.allclose(a_cluster.var.XY, b_cluster.var.XY)
    assert np.allclose(a_cluster.XY    , b_cluster.XY    )

    assert             a_cluster.nsipm == exactly(b_cluster.nsipm)
    assert             a_cluster.Q     == approx (b_cluster.Q    )
    assert             a_cluster.X     == approx (b_cluster.X    )
    assert             a_cluster.Y     == approx (b_cluster.Y    )
    assert             a_cluster.Xrms  == approx (b_cluster.Xrms )
    assert             a_cluster.Yrms  == approx (b_cluster.Yrms )
    assert             a_cluster.R     == approx (b_cluster.R    )
    assert             a_cluster.Phi   == approx (b_cluster.Phi  )

def assert_bhit_equality(a_hit, b_hit):
    assert np.allclose(a_hit.pos , b_hit.pos)
    assert np.allclose(a_hit.XYZ , b_hit.XYZ)

    assert  a_hit.E  == approx (b_hit.E)
    assert  a_hit.X  == approx (b_hit.X)
    assert  a_hit.Y  == approx (b_hit.Y)
    assert  a_hit.Z  == approx (b_hit.Z)

def assert_MChit_equality(a_hit, b_hit):
    assert_bhit_equality   (a_hit, b_hit)
    assert  a_hit.time  == approx (b_hit.time)

def assert_hit_equality(a_hit, b_hit):
    assert_bhit_equality   (a_hit, b_hit)
    assert_cluster_equality(a_hit, b_hit)
    assert a_hit.Ec           == approx (b_hit.Ec         )
    assert a_hit.Xpeak        == approx (b_hit.Xpeak      )
    assert a_hit.Ypeak        == approx (b_hit.Ypeak      )
    assert a_hit.peak_number  == exactly(b_hit.peak_number)
