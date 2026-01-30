import numpy as np
import pandas as pd
import tables

from invisible_cities.core.core_functions import in_range, fix_random_seed
from krmap_functions import create_empty_map, get_median, gaussian_fit, fit_map, merge_maps, include_coordinates, gauss_seed, quick_gauss_fit, create_time_slices, get_time_evol, save_map, compute_metadata, append_time_evol
from pytest import raises
from invisible_cities.core.fit_functions import fit, gauss, expo
from invisible_cities.types.symbols import SelRegionMethod



def test_create_empty_map_shape():
    xy_range = (-100, 100)
    dt_range = (20, 100)
    xy_nbins = 8
    dt_nbins = 5

    empty_map_test = create_empty_map(xy_range = xy_range,
                                      dt_range = dt_range,
                                      xy_nbins = xy_nbins,
                                      dt_nbins = dt_nbins)

    rows_test = dt_nbins * xy_nbins * xy_nbins
    n_columns_test = 8

    assert empty_map_test.shape == (rows_test, n_columns_test)



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



def test_get_median_empty_input():
    empty_dst = pd.DataFrame(columns = ['DT', 'x', 'y', 'S2e'])
    result = get_median(empty_dst)

    assert result.shape == (1, 5)
    assert result['nevents'].sum() == 0
    for col in ['mu', 'sigma', 'mu_error', 'sigma_error']:
        assert pd.isna(result[col].iloc[0]), f"{col} is not a nan"


def test_fitfun_medfun_Nevents():
    d = {'DT': np.empty(6), 'x' : np.empty(6), 'y' : np.empty(6),'S2e' : np.ones(6)}
    df_test = pd.DataFrame(data = d, index = range(0, 6))
    N = len(df_test)
    result_med = med_fun(df_test)
    assert result_med['nevents'].iloc[0] == N
    assert fit_function(df_test, bins = 6)['nevents'].iloc[0] == N


def test_merge_maps():
    Nan_map_test = create_empty_map(xy_range = (-500, 500), dt_range = (0, 1400), xy_nbins = 100, dt_nbins = 10)
    d = {'nevents': np.empty(6, dtype = int), 'mu' : np.empty(6), 'sigma' : np.empty(6), 'mu_error' : np.empty(6), 'sigma_error' : np.empty(6)}
    map_3D_test = pd.DataFrame(data = d, index = range(0, 6))
    assert merge_maps(Nan_map_test, map_3D_test).shape == Nan_map_test.shape



def test_medfun_works_with_even_data():
    d = {'DT': np.empty(6), 'x' : np.empty(6), 'y' : np.empty(6),'S2e' : [8000, 7500, 8300, 7900, 9000, 8100]}
    df_test = pd.DataFrame(data = d, index = range(0,6))
    S2e = df_test['S2e']
    result_med_fun = med_fun(df_test)
    result_med_fun = result_med_fun['median']
    result_med_data = 8050

    assert (result_med_fun == result_med_data).all()

def test_medfun_works_with_odd_data():
    d = {'DT': np.empty(7), 'x' : np.empty(7), 'y' : np.empty(7),'S2e' : [8000, 7500, 8300, 7900, 9000, 8100, 9100]}
    df_test = pd.DataFrame(data = d, index = range(0,7))
    S2e = df_test['S2e']
    result_med_fun = med_fun(df_test)
    result_med_fun = result_med_fun['median']
    result_med_data = 8100
    assert (result_med_fun == result_med_data).all()



def test_gaussianfit_computes_right_values(): #If the events follow a gaussian distribution, the median, mean and fit mean values should be the same

    with fix_random_seed(42):
        S2e_df = pd.DataFrame({
            'S2e': np.random.normal(loc = 8000, scale = 10.0, size = 10000)
        })

    results = gaussian_fit(S2e_df, bins = 50)
    results_median = get_median(S2e_df)

    assert np.isclose(results['mu'][0], S2e_df.S2e.mean(), atol = 1)
    assert np.isclose(results['mu'][0], results_median.median, atol = 1)
    assert np.isclose(S2e_df.S2e.mean(), results_median.median, atol = 1)
    assert np.isclose(results['sigma'][0], S2e_df.S2e.std(), atol = 0.5)
    assert np.isclose(results['mu_error'][0], S2e_df.S2e.std()/np.sqrt(len(S2e_df)), atol = 0.1)
    #error sigma?
    
    
 

def test_gauss_seed():

    x = np.linspace(1000, 1500, 100)

    amp_true = 1000
    mu_true = 1250
    sigma_true = 50

    y =gauss(x, amp_true, mu_true, sigma_true)
    amp_seed, x_max_seed, sigma_seed = gauss_seed(x, y)


    assert np.isclose(amp_seed, amp_true, atol = 5)
    assert np.isclose(x_max_seed, mu_true, atol = 1)
    assert np.isclose(sigma_seed, sigma_true, atol = 0.5)

