import numpy as np
import pandas as pd
from pytest import raises, fixture
import tables

from invisible_cities.core.fit_functions import fit, gauss, expo
from invisible_cities.types.symbols import SelRegionMethod
from invisible_cities.core.core_functions import in_range, fix_random_seed
from invisible_cities.core.testing_utils import assert_dataframes_close

from invisible_cities.icaros.krmap_functions import create_empty_map, get_median, gaussian_fit, fit_map, merge_maps, include_coordinates, gauss_seed, quick_gauss_fit, save_map, compute_metadata, get_time_evol_single_slice, get_time_evol, compute_3D_map


@fixture
def dummy_empty_df():
    return pd.DataFrame(columns = ['DT', 'x', 'y', 'S2e'])



@fixture
def dummy_get_median():
    return pd.DataFrame({'S2e' : [8000, 7500, 8300, 7900, 9000, 8100]})




@fixture
def dummy_include_coordinates():
    return  pd.DataFrame({'k':np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]),
                         'i':np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1]),
                         'j':np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]),
                         'nevents': np.empty(11, dtype = int),
                         'mu':np.empty(11),
                         'sigma' : np.empty(11),
                         'mu_error' : np.empty(11),
                         'sigma_error' : np.empty(11)}, index = range(0, 11))

@fixture
def dummy_get_time_evol():
    return pd.DataFrame({'time': np.linspace(0, 3599, 1001),
                      'S2e':np.linspace(7500, 8500, 1001),
                      'X': np.linspace(-100, 100, 1001),
                      'Y': np.linspace(-100, 100, 1001),
                      'DT': np.linspace(20, 1350, 1001),
                      'S1e':np.linspace(7, 10, 1001),
                      'S1h': np.linspace(1,2,1001),
                      'S1w': np.linspace(210, 240, 1001),
                      'Nsipm':np.linspace(10, 30, 1001),
                      'Xrms':np.linspace(13, 15, 1001),
                      'Yrms':np.linspace(13, 15, 1001),
                      'Zrms':np.linspace(3, 5, 1001),
                      'S2q':np.linspace(540, 600, 1001),
                      'S2w':np.linspace(20, 24, 1001),
                      'Ec': np.linspace(35, 45, 1001),
                      'Ec_2':np.random.normal(loc = 41.5, scale = 10, size = 1001)
                     })




def test_gauss_seed():

    x = np.linspace(1000, 1500, 100)

    amp_true = 1000
    mu_true = 1250
    sigma_true = 0.02 * mu_true

    y =gauss(x, amp_true, mu_true, sigma_true)
    amp_seed, x_max_seed, sigma_seed = gauss_seed(x, y)


    assert np.isclose(amp_seed, amp_true, atol = 10)
    assert np.isclose(x_max_seed, mu_true, atol = 1)
    assert np.isclose(sigma_seed, sigma_true, atol = 0.5)




def test_create_empty_map_shape():
    xy_range = (-100, 100)
    dt_range = (20, 100)
    xy_nbins = 8
    dt_nbins = 5

    empty_map  = create_empty_map(xy_range = xy_range,
                                      dt_range = dt_range,
                                      xy_nbins = xy_nbins,
                                      dt_nbins = dt_nbins)

    nrows  = dt_nbins * xy_nbins * xy_nbins
    n_columns = 8

    assert empty_map.shape == (nrows, n_columns)



def test_create_empty_map_values():
    xy_range = (-100, 100)
    dt_range = (20, 1000)
    xy_nbins = 8
    dt_nbins = 5

    empty_map_test = create_empty_map(xy_range = xy_range,
                                      dt_range = dt_range,
                                      xy_nbins = xy_nbins,
                                      dt_nbins = dt_nbins)

    assert empty_map_test.k.nunique() == dt_nbins
    assert empty_map_test.i.nunique() == xy_nbins
    assert empty_map_test.j.nunique() == xy_nbins

    data_columns = ['nevents', 'mu', 'sigma', 'mu_error', 'sigma_error']

    assert empty_map_test[data_columns].isna().all().all()



def test_get_median_empty_input_does_not_raise(dummy_empty_df):
    get_median(dummy_empty_df.S2e)



def test_get_median_empty_input(dummy_empty_df):
    result = get_median(dummy_empty_df.S2e)

    assert result.shape == (1, 5)
    assert result['nevents'].sum() == 0
    for col in ['mu', 'sigma', 'mu_error', 'sigma_error']:
        assert pd.isna(result[col].iloc[0]), f"{col} is not a nan"




def test_get_median_works_with_even_data(dummy_get_median):

    result = get_median(dummy_get_median.S2e)['mu']
    median_data = 8050

    assert (result == median_data).all()



def test_get_median_works_with_odd_data(dummy_get_median):

    dummy_get_median.loc[len(dummy_get_median.S2e)] = [9100]

    result_med_fun = get_median(dummy_get_median.S2e)
    result_med_fun = result_med_fun['mu']
    result_med_data = 8100

    assert (result_med_fun == result_med_data).all()




def test_get_median_errors():
    S2e_test1 = pd.DataFrame([1, 2, 3],
                             columns = ['S2e'])

    S2e_test2 = pd.DataFrame([1, 1, 1, 1,
                              2, 2, 2, 2,
                              3, 3, 3, 3],
                             columns = ['S2e'])

    map_test1  = get_median(S2e_test1.S2e)
    map_test2  = get_median(S2e_test2.S2e)

    ratio_error_values  = map_test1.mu_error.values / map_test2.mu_error.values
    ratio_sqrt = np.sqrt(len(S2e_test2)-1)/np.sqrt(len(S2e_test1)-1)

    assert np.allclose(map_test1.mu.values, map_test2.mu.values)
    assert np.isclose(ratio_error_values, ratio_sqrt)




def test_gaussian_fit_few_entries(dummy_empty_df):

    result_fit = gaussian_fit(dummy_empty_df, 1, min_events = 10)
    result_fun = get_median(dummy_empty_df.S2e)['mu']

    assert np.allclose(result_fit.mu.values, result_fun.values, equal_nan=True)




def test_gaussian_fit_Nevents():
    d = {'DT': np.empty(6), 'x' : np.empty(6), 'y' : np.empty(6),'S2e' : np.ones(6)}
    df_test = pd.DataFrame(data = d, index = range(0, 6))
    N = len(df_test)
    result_fit = gaussian_fit(df_test, nbins_S2e = 6)

    assert result_fit.nevents.iloc[0] == N




def test_gaussian_fit_computes_right_values():

    with fix_random_seed(42):
        S2e_df = pd.DataFrame({
            'S2e': np.random.normal(loc = 8000, scale = 10.0, size = 10000)
        })

    results = gaussian_fit(S2e_df, nbins_S2e = 50)

    assert np.isclose(results.mu[0], S2e_df.S2e.mean(), atol = 1)
    assert np.isclose(results.sigma[0], S2e_df.S2e.std(), atol = 0.5)
    assert np.isclose(results.mu_error[0], S2e_df.S2e.std()/np.sqrt(len(S2e_df)), atol = 0.1)
    assert np.isclose(results.sigma_error[0], 0.068, atol = 0.1)




def test_fit_map_S2e_values():
    xy_nbins = 4
    dt_nbins = 2
    xy_range = (-10, 10)
    dt_range = (20, 50)

    df = []

    with fix_random_seed(42):
        S2e = np.random.normal(loc = 8000, scale = 10.0, size = 10000)

    for i in range(xy_nbins):
        for j in range(xy_nbins):
            for k in range(dt_nbins):
                x_val = (i + 0.5) * (xy_range[1] - xy_range[0]) / xy_nbins
                y_val = (j + 0.5) * (xy_range[1] - xy_range[0]) / xy_nbins
                dt_val = (k + 0.5) * (dt_range[1] - dt_range[0]) / dt_nbins


                S2e_df = pd.DataFrame({
                        'X': np.full(10000, x_val),
                        'Y': np.full(10000, y_val),
                        'DT': np.full(10000, dt_val),
                        'S2e': S2e
                         })

                df.append(S2e_df)

    df  = pd.concat(df, ignore_index = True)

    result = fit_map(df = df,
                          xy_range = xy_range,
                          dt_range = dt_range,
                          xy_nbins = xy_nbins,
                          dt_nbins = dt_nbins,
                          fit_function = gaussian_fit,
                          nbins_S2e = 25,
                          S2e_range = (1000, 20000))

    reference = result.iloc[0]


    assert np.allclose(result.mu , reference.mu)
    assert np.allclose(result.sigma, reference.sigma)
    assert np.allclose(result.mu_error, reference.mu_error)
    assert np.allclose(result.sigma_error, reference.sigma_error)
    assert (result.nevents == reference.nevents).all()



def test_fit_map_missing_bins():
    """
    We are taking only half of the range for each variable (x and y from 0 to 100 and
    dt from 30 to 60) so the number of rows of the final dataframe should be the product
    of half of the number of bins (3*3*3 instead of 6*6*6)
    """
    xy_nbins = 6
    dt_nbins = 6
    xy_range = (-100, 100)
    dt_range = (20, 80)


    with fix_random_seed(42):
        S2e = np.random.normal(loc = 8000, scale = 10.0, size = 10)


        df       =  pd.DataFrame({
                    'X': np.random.uniform(0, 100, 100000),
                    'Y': np.random.uniform(0, 100, 100000),
                    'DT': np.random.uniform(30, 60, 100000),
                    'S2e' : np.random.normal(loc = 8000, scale = 10.0, size = 100000)})


    result = fit_map(df = df,
                          xy_range = xy_range,
                          dt_range = dt_range,
                          xy_nbins = xy_nbins,
                          dt_nbins = dt_nbins,
                          fit_function = gaussian_fit,
                          nbins_S2e = 25,
                          S2e_range = (1000, 20000))


    assert result.shape[0] == 3*3*3




def test_merge_maps():
    Nan_map_test = create_empty_map(xy_range = (-500, 500), dt_range = (0, 1400), xy_nbins = 100, dt_nbins = 10)
    #creating a map with 3 wholes to make sure that the shape still the same as NaNmaps
    d = {'k':np.array([1, 1, 1, 1, 2, 2, 2, 3, 3]),
         'i':np.array([1, 1, 2, 2, 1, 2, 2, 1, 2]),
         'j':np.array([1, 2, 1, 2, 1, 1, 2, 2, 1]),
         'nevents': np.empty(9, dtype = int),
         'mu' : np.empty(9),
         'sigma' : np.empty(9),
         'mu_error' : np.empty(9),
         'sigma_error' : np.empty(9)}

    map_3D_test = pd.DataFrame(data = d, index = range(0, 9))
    assert merge_maps(Nan_map_test, map_3D_test).shape == Nan_map_test.shape



def test_merge_maps_empty():
    Nan_map = create_empty_map(xy_range = (-500, 500), dt_range = (0, 1400), xy_nbins = 100, dt_nbins = 10)
    #creating am empty map to make sure it works
    d = {'k':np.full(9, 0),
         'i':np.full(9, 0),
         'j':np.full(9, 0),
         'nevents': np.empty(9, dtype = int),
         'mu' : np.empty(9),
         'sigma' : np.empty(9),
         'mu_error' : np.empty(9),
         'sigma_error' : np.empty(9)}

    map_3D = pd.DataFrame(data = d, index = range(0, 9))

    assert merge_maps(Nan_map, map_3D).shape == Nan_map.shape



def test_include_coordinates_shape(dummy_include_coordinates):
    xy_range = (-100, 100)
    dt_range = (20, 100)
    xy_nbins = 10
    dt_nbins  = 5

    full_map = include_coordinates(dummy_include_coordinates, xy_range = xy_range, dt_range = dt_range, xy_nbins = xy_nbins, dt_nbins = dt_nbins)

    assert full_map.shape[0] == dummy_include_coordinates.shape[0]
    assert full_map.shape[1] == dummy_include_coordinates.shape[1] + 3




def test_include_coordinates_range(dummy_include_coordinates):

    xy_range = (-100, 100)
    dt_range = (20, 100)
    xy_nbins = 10
    dt_nbins  = 5

    Nan_map_test = create_empty_map(xy_range = xy_range, dt_range = dt_range, xy_nbins = xy_nbins, dt_nbins = dt_nbins)

    krmap_test = merge_maps(Nan_map_test, dummy_include_coordinates)

    full_map = include_coordinates(krmap_test, xy_range = xy_range, dt_range = dt_range, xy_nbins = xy_nbins, dt_nbins = dt_nbins)


    assert np.all(in_range(full_map.x, *xy_range))
    assert np.all(in_range(full_map.y, *xy_range))
    assert np.all(in_range(full_map.dt, *dt_range))




def test_include_coordinates_bincenter(dummy_include_coordinates):

    xy_range = (-100, 100)
    dt_range = (20, 100)

    full_map = include_coordinates(dummy_include_coordinates, xy_range = xy_range, dt_range = dt_range, xy_nbins = 1, dt_nbins = 1)

    x_center = xy_range[0] + 0.5*(xy_range[1] - xy_range[0])
    dt_center = dt_range[0] + 0.5*(dt_range[1] - dt_range[0])

    assert np.isclose(full_map.x.min(), x_center)
    assert np.isclose(full_map.dt.min(), dt_center)


def test_compute_3D_map():
    with fix_random_seed(42):
        df_ = pd.DataFrame({'X' : np.random.uniform(-100, 100, 1000000),
                        'Y' : np.random.uniform(-100, 100, 1000000),
                        'DT' : np.random.uniform(20, 1350, 1000000),
                       'S2e' : np.random.normal(8000, 150, 1000000)
                      })

    full_map = compute_3D_map(df_, xy_range = (-100, 100), dt_range = (20, 1350), xy_nbins = 10, dt_nbins = 10, fit_function = gaussian_fit, nbins_S2e = 50, S2e_range = (7000, 9000))

    assert np.allclose(full_map.mu, 8000, atol = 30)
    assert np.allclose(full_map.sigma, 150, atol = 20)
    assert (full_map.mu_error < 30).all()



def test_compute_metadata_single_column():

    xy_range = (-100, 100)
    dt_range = (40, 100)
    xy_nbins = 1
    dt_nbins = 1

    df_test = pd.DataFrame({'R': np.linspace(0, 40, 3),
                            'Z': np.linspace(40, 100, 3)
                            })

    full_map_test = pd.DataFrame({'dt': np.linspace(40, 100, 3),
                                  'x': np.linspace(-100, 100, 3),
                                  'y': np.linspace(-100, 100, 3),
                                  'k':np.linspace(1, 3, 3),
                                  'i':np.linspace(1, 3, 3),
                                  'j':np.linspace(1, 3, 3),
                                  'nevents': np.linspace(1, 3, 3),
                                  'mu' : np.linspace(1, 3, 3),
                                  'sigma' : np.linspace(1, 3, 3),
                                  'mu_error' : np.linspace(1, 3, 3),
                                  'sigma_error' : np.linspace(1, 3, 3)})

    metadata_test = compute_metadata(df_test,
                                     full_map_test,
                                     xy_range = xy_range,
                                     dt_range = dt_range,
                                     xy_nbins = xy_nbins,
                                     dt_nbins = dt_nbins)

    assert metadata_test.shape[1] == 1





def test_quick_gauss_fit():
    with fix_random_seed(42):
        x = np.random.normal(loc = 8000, scale = 10, size = 1000)
    f = quick_gauss_fit(x, bins = 10, sigma = None)

    assert np.isclose(x.mean(), f.values[1], atol = 1)
    assert np.isclose(x.std(), f.values[2], atol = 1)



def test_get_time_evol_single_slice_shape(dummy_get_time_evol):

   t_evol = get_time_evol_single_slice(dummy_get_time_evol, 'Ec_2', 1, 0, 0, SelRegionMethod.circle, 1000, np.linspace(1250, 1400, 50), (1000, 1350), np.linspace(30, 60, 101))

   assert t_evol.shape[1] == 33
   assert t_evol.shape[0] == 1



def test_get_time_evol(dummy_get_time_evol):

    df2 = pd.concat([dummy_get_time_evol,
                     dummy_get_time_evol.assign(time = dummy_get_time_evol.time + 3600),
                     dummy_get_time_evol.assign(time= dummy_get_time_evol.time + 7200)],
                     ignore_index = True)

    t_evols = get_time_evol(df2, 1, 'Ec_2', 1, 0, 0, SelRegionMethod.circle, 1000, np.linspace(1250, 1400, 50), (1000, 1350), np.linspace(30, 60, 101), error = False)

    assert len(df2) == 3003
    for col in t_evols.columns:
        if col == 'ts':
            continue
        assert np.allclose(t_evols[col], t_evols[col][0])



def test_save_map():
    krmap = pd.DataFrame({'k':np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]),
                          'i':np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1]),
                          'j':np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]),
                          'nevents': np.empty(11, dtype = int),
                          'mu' : np.empty(11),
                          'sigma' : np.empty(11),
                          'mu_error' : np.empty(11),
                          'sigma_error' : np.empty(11),
                          'dt': np.linspace(20, 1350, 11),
                          'x': np.linspace(-450, 450, 11),
                          'y': np.linspace(-450, 450, 11),
                         })

    eff = pd.DataFrame({'eff_diffusion_band': 0.8759,
                                 'eff_Xrays': 0.8039,
                                 'eff_1S1_1S2': 0.8377,
                                 'eff_S2_trigger_time': 0.9970,
                                 'eff_Rmax': 0.9859,
                                 'eff_range_DT': 0.9951,
                                 'eff_NSiPMS': 1.0,
                                 'total_efficiency': 0.5770
                                 }, index = [0])


    meta = {'rmax'       : 100,
            'zmax'        : 1000,
            'bin_size_dt' : 5,
            'bin_size_x'  : 4,
            'bin_size_y'  : 4,
            'dtbins'      : [(0, 1, 2, 3, 4)],
            'xbins'       : [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)],
            'ybins'       : [(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)],
            'nbins_dt'    : 5,
            'nbins_x'     : 10,
            'nbins_y'     : 10,
            'xy_range'    : [(-100, 100)],
            'dt_range'    : [(20, 1350)],
            'map_shape'   : [(11, 11)],
            'map_extent'  : 11}

    meta = pd.DataFrame(meta, index = [0]).T


    t_evolution = pd.DataFrame({'run_number' : 1,
              'ts' :1e6,
              's2e': 8000,
              's2eu' : 10,
              'ec' : 41.5,
              'ecu': 0.5,
              'chi2_ec': 1,
              'e0': 7900,
              'e0u': 5,
              'lt' : 30000 ,
              'ltu' : 5,
              'dv' : 0.91308,
              'dvu' : 0.01,
              'resol' :4.01,
              'resolu' : 0.146,
              's1w' : 225,
              's1wu' : 0.87,
              's1e' : 8,
              's1eu' :0.3,
              's1h': 100,
              's1hu':1,
              's2w': 8,
              's2wu': 4,
              's2q': 570.37,
              's2qu': 0.37,
              'Nsipm': 16.88,
              'Nsipmu': 0.005,
              'Xrms': 14.20,
              'Xrmsu': 0.004,
              'Yrms': 14.15,
              'Yrmsu':0.004 ,
              'Zrms': 4.15,
              'Zrmsu': 0.003}, index = [0])

    save_map('testmap.h5', eff, krmap, meta, t_evolution)

    krmap_3D = pd.read_hdf('testmap.h5', 'krmap/krmap')
    efficiencies = pd.read_hdf('testmap.h5', 'data/selection_efficiencies')
    metadata = pd.read_hdf('testmap.h5')
    t_evol = pd.read_hdf('testmap.h5', 't_evol/t_evol')

    testmap = tables.open_file('testmap.h5', mode = 'r')


    assert_dataframes_close(krmap_3D, krmap)
    assert_dataframes_close(efficiencies, eff)
    assert_dataframes_close(metadata,meta)
    assert_dataframes_close(t_evol,t_evolution)
