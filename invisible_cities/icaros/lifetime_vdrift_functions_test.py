import pandas as pd
import numpy  as np

from lifetime_vdrift_functions import select_lifetime_region
from invisible_cities.types.symbols import SelRegionMethod
from invisible_cities.core.core_functions import in_range, fix_random_seed


def test_select_lifetime_region_square():
    shape_size = 30

    with fix_random_seed(42):
        df = pd.DataFrame({'X': np.random.uniform(-100, 100, 1000),
                           'Y': np.random.uniform(-100, 100, 1000)})

    df_sel = select_lifetime_region(df, 0, 0, shape = SelRegionMethod.square, shape_size = 30)


    assert in_range(df_sel.X, -shape_size, shape_size).all()
    assert in_range(df_sel.Y, -shape_size, shape_size).all()




def test_select_lifetime_region_circle():
    shape_size = 30

    with fix_random_seed(42):
        df = pd.DataFrame({'X': np.random.uniform(-100, 100, 1000),
                           'Y': np.random.uniform(-100, 100, 1000)})

    df_sel = select_lifetime_region(df, 0, 0, shape = SelRegionMethod.circle, shape_size = 30)

    assert ((df_sel.X**2 + df_sel.Y**2) <= shape_size**2).all()




def test_select_lifetime_region_values_square():

    x = np.linspace(0, 10, 11)

    df = pd.DataFrame(data =
                           {'X': x,
                            'Y': -x
                            })

    df_sel = select_lifetime_region(df, 5, -5, shape = SelRegionMethod.square, shape_size = 2.5)

    assert in_range(df_sel.X, 3, 8).all()
    assert in_range(df_sel.Y, -7, -2).all()




def test_select_lifetime_region_values_circle():

    x = np.linspace(0, 10, 11)

    df = pd.DataFrame(data =
                          {'X': x,
                           'Y': -x
                          })
    df_sel = select_lifetime_region(df, 5, -5, shape = SelRegionMethod.circle, shape_size = 2.5)

    #df_sel would include only those values of x and y that satisfy (x-5)**2 + (y+5)**2 <= r**2

    assert in_range(df_sel.X, 4, 7).all()
    assert in_range(df_sel.Y, -6, -3).all()
