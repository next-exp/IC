import pandas as pd
import numpy  as np
import warnings
from pytest import fixture

from lifetime_vdrift_functions import select_lifetime_region, compute_drift_v
from invisible_cities.types.symbols import SelRegionMethod
from invisible_cities.core.core_functions import in_range, fix_random_seed

@fixture
def dummy_lifetime_df():
    with fix_random_seed(42):
        df = pd.DataFrame({'X': np.random.uniform(-100, 100, 1000),
                           'Y': np.random.uniform(-100, 100, 1000)})
    return df

@fixture
def dummy_lifetime_limits():
    x = np.linspace(0, 10, 11)
    df = pd.DataFrame({'X': x,
                       'Y': -x})
    return df




def test_select_lifetime_region_square(dummy_lifetime_df):

    shape_size = 30
    df_sel = select_lifetime_region(dummy_lifetime_df, 0, 0, shape = SelRegionMethod.square, shape_size = 30)

    assert in_range(df_sel.X, -shape_size, shape_size).all()
    assert in_range(df_sel.Y, -shape_size, shape_size).all()



def test_select_lifetime_region_circle(dummy_lifetime_df):

    shape_size = 30
    df_sel = select_lifetime_region(dummy_lifetime_df, 0, 0, shape = SelRegionMethod.circle, shape_size = 30)

    assert ((df_sel.X**2 + df_sel.Y**2) <= shape_size**2).all()




def test_select_lifetime_limit_values_square(dummy_lifetime_limits):
    df_sel = select_lifetime_region(dummy_lifetime_limits, 5, -5, shape = SelRegionMethod.square, shape_size = 2.5)

    assert in_range(df_sel.X, 3, 8).all()
    assert in_range(df_sel.Y, -7, -2).all()



def test_select_lifetime_region_values_square(dummy_lifetime_limits):
    df_sel = select_lifetime_region(dummy_lifetime_limits, 5, -5, shape = SelRegionMethod.square, shape_size = 2.5)

    assert (df_sel.X == [3, 4, 5, 6, 7]).all()
    assert (df_sel.Y == [-3, -4, -5, -6, -7]).all()




def test_select_lifetime_limit_values_circle(dummy_lifetime_limits):
    df_sel = select_lifetime_region(dummy_lifetime_limits, 5, -5, shape = SelRegionMethod.circle, shape_size = 2.5)

    #df_sel would include only those values of x and y that satisfy (x-5)**2 + (y+5)**2 <= r**2
    assert in_range(df_sel.X, 4, 7).all()
    assert in_range(df_sel.Y, -6, -3).all()



def test_select_lifetime_region_values_circle(dummy_lifetime_limits):
    df_sel = select_lifetime_region(dummy_lifetime_limits, 5, -5, shape = SelRegionMethod.circle, shape_size = 2.5)

    #df_sel would include only those values of x and y that satisfy (x-5)**2 + (y+5)**2 <= r**2
    assert (df_sel.X == [4, 5, 6]).all()
    assert (df_sel.Y == [-4, -5, -6]).all()


def test_compute_drift_v_positive(): #dv > 0 as long as inflection > 0

    dtdata = np.linspace(20, 1350, 1000)
    dtbins = np.linspace(1200, 1500, 100)

    dv, dvu = compute_drift_v(dtdata, dtbins, seed = None)

    assert dv > 0


def test_compute_drift_v_zero_inflection(): #if fit fails, dv = nan

    dtdata = np.linspace(-1, 1, 101)
    dtbins = np.linspace(-1, 1, 101)

    seed = (1, 0, 1, 0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dv, dvu = compute_drift_v(dtdata, dtbins, seed = seed)

    assert np.isnan(dv)
    assert np.isnan(dvu)
