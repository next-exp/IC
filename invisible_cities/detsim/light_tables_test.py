import os
import numpy as np

from .. core.core_functions import find_nearest

from .. io.dst_io  import load_dst

from .light_tables import create_lighttable_function

from pytest import fixture

from hypothesis import given, settings
from hypothesis.strategies  import floats
from hypothesis.extra.numpy import arrays


@fixture(scope='session')
def lighttable_filenames(ICDATADIR):
    s1ltfname = os.path.join(ICDATADIR, 'NEXT_NEW.energy.S1.PmtR11410.LightTable.h5')
    s2ltfname = os.path.join(ICDATADIR, 'NEXT_NEW.energy.S2.PmtR11410.LightTable.h5')
    return  {'s1': s1ltfname,
             's2': s2ltfname}


@given(xs=floats(min_value=-500, max_value=500),
       ys=floats(min_value=-500, max_value=500),
       zs=floats(min_value=-500, max_value=500))
def test_get_lt_values_s1(lighttable_filenames, xs, ys, zs):

    s1_lighttable = lighttable_filenames["s1"]
    S1_LT = create_lighttable_function(s1_lighttable)

    lt     = load_dst(s1_lighttable, "LT", "LightTable")
    Config = load_dst(s1_lighttable, "LT", "Config")    .set_index("parameter")
    sensor = Config.loc["sensor"].value
    act_r  = float(Config.loc["ACTIVE_rad"].value)
    lt     = lt.drop(sensor + "_total", axis=1) # drop total column
    lt     = lt.set_index(["x", "y", "z"])

    x_lt = lt.index.get_level_values("x").unique()
    y_lt = lt.index.get_level_values("y").unique()
    z_lt = lt.index.get_level_values("z").unique()

    if (np.sqrt(xs**2 + ys**2)<=act_r) & (z_lt.min()<=zs) & (zs<=z_lt.max()):
        xnearest = find_nearest(x_lt, xs)
        ynearest = find_nearest(y_lt, ys)
        znearest = find_nearest(z_lt, zs)
        expected = lt.loc[xnearest, ynearest, znearest].values[None, ...]
    else:
        expected = np.zeros((1, 12))
    np.testing.assert_allclose(S1_LT(np.array([xs]), np.array([ys]), np.array([zs])), expected)


@given(xs=floats(min_value=-500, max_value=500),
       ys=floats(min_value=-500, max_value=500))
def test_get_lt_values_s2(lighttable_filenames, xs, ys):

    s2_lighttable = lighttable_filenames["s2"]
    S2_LT = create_lighttable_function(s2_lighttable)

    lt     = load_dst(s2_lighttable, "LT", "LightTable")
    Config = load_dst(s2_lighttable, "LT", "Config")    .set_index("parameter")
    sensor = Config.loc["sensor"].value
    act_r  = float(Config.loc["ACTIVE_rad"].value)
    lt     = lt.drop(sensor + "_total", axis=1) # drop total column
    lt     = lt.set_index(["x", "y"])

    x_lt = lt.index.get_level_values("x").unique()
    y_lt = lt.index.get_level_values("y").unique()

    if (np.sqrt(xs**2 + ys**2)<=act_r):
        xnearest = find_nearest(x_lt, xs)
        ynearest = find_nearest(y_lt, ys)
        expected = lt.loc[xnearest, ynearest].values[None, ...]
    else:
        expected = np.zeros((1, 12))

    np.testing.assert_allclose(S2_LT(np.array([xs]), np.array([ys])), expected)
