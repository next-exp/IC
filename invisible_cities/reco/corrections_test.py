import os
import numpy  as np
import tables as tb

from . corrections import maps_coefficient_getter
from . corrections import read_maps
from . corrections import ASectorMap
from . corrections import correct_geometry_
from . corrections import correct_lifetime_
from . corrections import time_coefs_corr
from . corrections import get_df_to_z_converter
from . corrections import get_normalization_factor
from . corrections import apply_all_correction_single_maps
from . corrections import apply_all_correction

from pytest                import fixture
from numpy.testing         import assert_allclose
from numpy.testing         import assert_array_equal
from numpy.testing         import assert_raises

from hypothesis.strategies import floats
from hypothesis.strategies import integers
from hypothesis.strategies import composite
from hypothesis.strategies import lists
from hypothesis            import given
from hypothesis            import settings

from ..types.symbols import NormStrategy

from invisible_cities.core.testing_utils import random_length_float_arrays
from invisible_cities.core.testing_utils import float_arrays
from invisible_cities.core.exceptions    import TimeEvolutionTableMissing
from invisible_cities.core               import system_of_units            as units


@fixture(scope='session')
def map_filename(ICDATADIR):
    test_file = "kr_emap_xy_constant_values.h5"
    test_file = os.path.join(ICDATADIR, test_file)
    return test_file

@fixture(scope='session')
def map_filename_MC(ICDATADIR):
    test_file = "kr_emap_xy_constant_values_RN_negat.h5"
    test_file = os.path.join(ICDATADIR, test_file)
    return test_file

@fixture
def toy_corrections(correction_map_filename):
    xs, ys = np.meshgrid(np.linspace(-199,199,5), np.linspace(-199,199,5))
    xs     = xs.flatten()
    ys     = ys.flatten()
    zs     = np.ones(25)

    ts         = np.array([1.54513805e+09, 1.54514543e+09, 1.54515280e+09, 1.54516018e+09,
                           1.54516755e+09, 1.54517493e+09, 1.54518230e+09, 1.54518968e+09,
                           1.54519705e+09, 1.54520443e+09, 1.54521180e+09, 1.54521918e+09,
                           1.54522655e+09, 1.54523393e+09, 1.54524130e+09, 1.54524868e+09,
                           1.54525605e+09, 1.54526343e+09, 1.54527080e+09, 1.54527818e+09,
                           1.54528555e+09, 1.54529293e+09, 1.54530030e+09, 1.54530768e+09,
                           1.54531505e+09])
    e0coef     = np.array([10632.51025668, 10632.51025668,  8198.00693508, 10632.51025668,
                           10632.51025668, 10632.51025668, 12083.6205191 , 12215.5218439 ,
                           11031.20482006, 10632.51025668, 10632.51025668, 12250.67984846,
                           12662.43500038, 11687.13986784,  7881.16756887, 10632.51025668,
                           11005.19041964, 11454.06577668, 10605.04377619, 10632.51025668,
                           10632.51025668, 10632.51025668, 10632.51025668, 10632.51025668,
                           10632.51025668])
    ltcoef     = np.array([3611.42910617, 3611.42910617, 2383.13016371, 3611.42910617,
                           3611.42910617, 3611.42910617, 4269.56836159, 4289.18987023,
                           4165.14681987, 3611.42910617, 3611.42910617, 3815.34850334,
                           3666.44326169, 3680.6402539 , 2513.42432537, 3611.42910617,
                           3552.99407017, 3514.83760875, 3348.95744382, 3611.42910617,
                           3611.42910617, 3611.42910617, 3611.42910617, 3611.42910617,
                           3611.42910617])
    correction = np.array([9.416792493e-05, 9.413238325e-05, 1.220993888e-04, 9.413193693e-05,
                           9.411739173e-05, 9.409713667e-05, 8.279152513e-05, 8.189972046e-05,
                           9.070379578e-05, 9.408888204e-05, 9.407901934e-05, 8.167086609e-05,
                           7.900813579e-05, 8.560232940e-05, 1.269748871e-04, 9.407975724e-05,
                           9.084074259e-05, 8.729552081e-05, 9.426779280e-05, 9.404039796e-05,
                           9.402650568e-05, 9.402980196e-05, 9.405365918e-05, 9.401772495e-05,
                           9.398598283e-05])

    return xs, ys, zs, ts, e0coef, ltcoef, correction


few_examples = settings(deadline=None, max_examples=100)

def test_maps_coefficient_getter_exact(toy_corrections, correction_map_filename):
    maps = read_maps(correction_map_filename)
    xs, ys, zs, _, coef_geo, coef_lt, _ = toy_corrections
    get_maps_coefficient_e0= maps_coefficient_getter(maps.mapinfo, maps.e0)
    CE  = get_maps_coefficient_e0(xs,ys)
    get_maps_coefficient_lt= maps_coefficient_getter(maps.mapinfo, maps.lt)
    LT  = get_maps_coefficient_lt(xs,ys)
    assert_allclose (CE, coef_geo)
    assert_allclose (LT, coef_lt)

def test_read_maps_returns_ASectorMap(correction_map_filename):
    maps = read_maps(correction_map_filename)
    assert type(maps)==ASectorMap

@composite
def xy_pos(draw, elements=floats(min_value=-250, max_value=250)):
    size = draw(integers(min_value=1, max_value=10))
    x    = draw(lists(elements,min_size=size, max_size=size))
    y    = draw(lists(elements,min_size=size, max_size=size))
    return (np.array(x),np.array(y))

@few_examples
@given(xy_pos = xy_pos())
def test_maps_coefficient_getter_gives_nans(correction_map_filename, xy_pos):
    x,y       = xy_pos
    maps      = read_maps(correction_map_filename)
    mapinfo   = maps.mapinfo
    map_df    = maps.e0
    xmin,xmax = mapinfo.xmin,mapinfo.xmax
    ymin,ymax = mapinfo.ymin,mapinfo.ymax
    get_maps_coefficient_e0= maps_coefficient_getter(mapinfo, map_df)
    CE        = get_maps_coefficient_e0(x,y)
    mask_x    = (x >=xmax) | (x<xmin)
    mask_y    = (y >=ymax) | (y<ymin)
    mask_nan  = (mask_x)   | (mask_y)
    assert all(np.isnan(CE[mask_nan]))
    assert not any(np.isnan(CE[~mask_nan]))

def test_read_maps_when_MC_t_evol_is_none(map_filename_MC):
    emap = read_maps(map_filename_MC)
    assert emap.t_evol is None

def test_read_maps_t_evol_table_is_correct(map_filename):
    """
    For this test, a map where its t_evol table correspond to single values
    (ts=[1,1,...1], e0=[2,2,...2], ..., Yrmsu=[27,27,...27])
    has been generated.
    """

    maps       = read_maps(map_filename)
    map_te     = maps.t_evol
    columns_te = map_te.columns
    assert np.all([np.all( map_te[parameter] == i+1 )
                   for i, parameter in zip( range(len(columns_te)), columns_te )])


def test_read_maps_maps_are_correct(map_filename):
    """
    For this test, a map where its correction maps are:
    (chi=[1,1,...1], e0=[13000,13000,...13000], e0u=[2,2,...2],
    lt=[5000,5000,...5000], ltu=[3,3,...3])
    has been generated.
    """

    maps = read_maps(map_filename)
    assert np.all(maps.chi2 == 1    )
    assert np.all(maps.e0   == 13000)
    assert np.all(maps.e0u  == 2    )
    assert np.all(maps.lt   == 5000 )
    assert np.all(maps.ltu  == 3    )
    assert maps.t_evol is not None


def test_read_maps_data_without_time_evolution(map_filename, config_tmpdir):
    tmp_filename = os.path.join(config_tmpdir, "map_without_tevol.h5")
    with tb.open_file(map_filename) as inputf:
        with tb.open_file(tmp_filename, "w") as outputf:
            for group in "chi2 e0 e0u lt ltu mapinfo".split():
                outputf.copy_node(getattr(inputf.root, group)
                                 , outputf.root
                                 , recursive = True)

    maps = read_maps(tmp_filename)
    assert np.all(maps.chi2 == 1    )
    assert np.all(maps.e0   == 13000)
    assert np.all(maps.e0u  == 2    )
    assert np.all(maps.lt   == 5000 )
    assert np.all(maps.ltu  == 3    )
    assert maps.t_evol is None


@given(random_length_float_arrays(min_value = 1e-4,
                                  max_value = 3e4 ))
def test_correct_geometry_properly(x):
    assert_array_equal(correct_geometry_(x),(1/x))


@given(float_arrays(min_value = 0,
                    max_value = 530),
       float_arrays(min_value = 1,
                    max_value = 1e4))
def test_correct_lifetime_properly(z, lt):
    compute_corr = np.exp(z / lt)
    assert_array_equal(correct_lifetime_(z, lt), compute_corr)

@few_examples
@given(floats(min_value = 0,
              max_value = 1e4))
def test_time_coefs_corr(map_filename, time):
    """
    In the map taken as input, none of the parameters
    changes with time, thus all outputs of
    time_coefs_corr function must be 1.
    """

    maps    = read_maps(map_filename)
    map_t   = maps.t_evol
    columns = map_t.columns
    # Error columns end with u
    err_columns   = [c for c in columns[1:]  if c[-1]=='u' ]
    value_columns = [c for c in  columns[1:] if c not in err_columns ]

    for col_value, col_err in zip(value_columns, err_columns):
        assert_allclose(time_coefs_corr(time, map_t['ts'], map_t[col_value], map_t[col_err]), 1)

@few_examples
@given(float_arrays(size      = 1000,
                    min_value = 0,
                    max_value = 550))
def test_get_df_to_z_converter_dv_known(map_filename, z):
    z           = np.array(z)
    map_z       = read_maps(map_filename)
    z_converter = get_df_to_z_converter(map_z)
    dv_mean     = np.mean(map_z.t_evol.dv)
    assert_array_equal(z_converter(z), z * dv_mean)

def test_get_df_to_z_converter_raises_exception_when_no_map(map_filename):
    assert_raises(TimeEvolutionTableMissing,
                  get_df_to_z_converter,
                  None)

def test_get_df_to_z_converter_raises_exception_when_invalid_map(map_filename_MC):
    map_e = read_maps(map_filename_MC)
    assert_raises(TimeEvolutionTableMissing,
                  get_df_to_z_converter,
                  map_e)

def test_get_normalization_factor_max_norm(correction_map_filename):
    map_e  = read_maps(correction_map_filename)
    factor = get_normalization_factor(map_e, NormStrategy.max)
    norm   = map_e.e0.max().max()
    assert factor == norm

def test_get_normalization_factor_mean_norm(correction_map_filename):
    map_e  = read_maps(correction_map_filename)
    factor = get_normalization_factor(map_e, NormStrategy.mean)
    norm   = np.mean(np.mean(map_e.e0))
    assert factor == norm

@few_examples
@given(floats(min_value = 1,
              max_value = 1e4))
def test_get_normalization_factor_custom_norm(correction_map_filename, custom_vaule):
    map_e  = read_maps(correction_map_filename)
    factor = get_normalization_factor(map_e,
                                      NormStrategy.custom,
                                      custom_vaule)
    norm   = custom_vaule
    assert factor == norm

def test_get_normalization_factor_krscale(correction_map_filename):
    map_e     = read_maps(correction_map_filename)
    factor    = get_normalization_factor(map_e, NormStrategy.kr)
    kr_energy = 41.5575 * units.keV
    assert factor == kr_energy

def test_get_normalization_factor_custom_norm_raises_exception_when_no_value(map_filename):
    map_e = read_maps(map_filename)
    assert_raises(ValueError,
                  get_normalization_factor,
                  map_e, NormStrategy.custom)

def test_get_normalization_factor_raises_exception_when_no_valid_norm(map_filename):
    map_e = read_maps(map_filename)
    assert_raises(ValueError,
                  get_normalization_factor,
                  map_e, None)

def test__apply_all_correction_single_maps_raises_exception_when_no_map(map_filename):
    map_e = read_maps(map_filename)
    assert_raises(TimeEvolutionTableMissing,
                  apply_all_correction_single_maps,
                  map_e, map_e, None)

def test_apply_all_correction_single_maps_raises_exception_when_invalid_map(map_filename_MC):
    map_e = read_maps(map_filename_MC)
    assert_raises(TimeEvolutionTableMissing,
                  apply_all_correction_single_maps,
                  map_e, map_e, map_e,
                  apply_temp = True)

@few_examples
@given(float_arrays(size      = 1,
                   min_value = -198,
                   max_value = +198),
       float_arrays(size      = 1,
                   min_value = -198,
                   max_value = +198),
       float_arrays(size      = 1,
                   min_value = 0,
                   max_value = 5e2),
       float_arrays(size      = 1,
                   min_value = 0,
                   max_value = 1e5))
def test_apply_all_correction_single_maps_properly(map_filename, x, y, z, t):
    """
    Due the map taken as input, the geometric correction
    factor must be 1, the temporal correction 1 and the
    lifetime one: exp(Z/5000).
    """
    maps      = read_maps(map_filename)
    load_corr = apply_all_correction_single_maps(map_e0     = maps,
                                                 map_lt     = maps,
                                                 map_te     = maps,
                                                 apply_temp = True,
                                                 norm_strat = NormStrategy.max)
    corr   = load_corr(x, y, z, t)
    result = np.exp(z/5000)
    assert_allclose (corr, result)

@few_examples
@given(float_arrays(size      = 1,
                    min_value = -198,
                    max_value = +198),
       float_arrays(size      = 1,
                    min_value = -198,
                    max_value = +198),
       float_arrays(size      = 1,
                    min_value = 0,
                    max_value = 5e2),
       float_arrays(size      = 1,
                    min_value = 0,
                    max_value = 1e5))
def test_correction_single_maps_equiv_to_all_correction(map_filename,
                                                        x, y, z, t):
    maps           = read_maps(map_filename)
    load_corr_uniq = apply_all_correction(maps, True)
    load_corr_diff = apply_all_correction_single_maps(map_e0     = maps,
                                                      map_lt     = maps,
                                                      map_te     = maps,
                                                      apply_temp = True)
    corr_uniq      = load_corr_uniq(x, y, z, t)
    corr_diff      = load_corr_diff(x, y, z, t)
    assert corr_uniq == corr_diff

def test_corrections_exact(toy_corrections, correction_map_filename):
    maps       = read_maps(correction_map_filename)
    xs, ys, zs, ts, _, _, factor = toy_corrections
    get_factor = apply_all_correction(maps       = maps,
                                      apply_temp = True,
                                      norm_strat = NormStrategy.custom,
                                      norm_value = 1.)
    fac        = get_factor(xs, ys, zs, ts)
    assert_allclose (factor, fac, atol = 1e-13)
