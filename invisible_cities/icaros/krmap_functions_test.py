import numpy as np
import pandas as pd
from invisible_cities.core.core_functions import in_range, shift_to_bin_centers
import itertools
from krmap_functions import create_empty_map, get_median, gaussian_fit, fit_map, merge_maps, include_coordinates, gauss_seed, quick_gauss_fit, create_time_slices, get_time_evol, save_map
from pytest import raises
from invisible_cities.core.core_functions import fix_random_seed



def test_medfun_empty_input():
    empty_dst = pd.DataFrame(columns = ['DT', 'x', 'y', 'S2e'])
    result = med_fun(empty_dst)

    assert result.shape == (1, 5)
    assert result['nevents'].sum() == 0
    for col in ['median', 'sigma', 'median_error', 'sigma_error']:
        assert pd.isna(result[col].iloc[0]), f"{col} is not a nan"



def test_medfun_empty_input_does_not_raise():
    empty_dst = pd.DataFrame(columns = ['DT', 'x', 'y', 'S2e'])

    #with raises(ValueError):
    med_fun(empty_dst)


def test_gaussianfit_does_fit_right():
    empty_dst = pd.DataFrame(columns = ['DT', 'x', 'y', 'S2e'])
    result_fit = gaussian_fit(empty_dst, 1)
    result_fun = med_fun(empty_dst)

    assert np.allclose(result_fit.mu.values, result_fun['median'].values, equal_nan=True)
    


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
    
    
 
