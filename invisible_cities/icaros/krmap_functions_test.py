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



def test_get_median_empty_input_does_not_raise():
    empty_dst = pd.DataFrame(columns = ['DT', 'x', 'y', 'S2e'])

    get_median(empty_dst)




def test_get_median_works_with_even_data():
    d = {'DT': np.empty(6), 'x' : np.empty(6), 'y' : np.empty(6),'S2e' : [8000, 7500, 8300, 7900, 9000, 8100]}
    df_test = pd.DataFrame(data = d, index = range(0,6))
    S2e = df_test['S2e']
    result = get_median(df_test)['mu']
    median_data = 8050

    assert (result == median_data).all()



def test_get_median_works_with_odd_data():
    d = {'DT': np.empty(7), 'x' : np.empty(7), 'y' : np.empty(7),'S2e' : [8000, 7500, 8300, 7900, 9000, 8100, 9100]}
    df_test = pd.DataFrame(data = d, index = range(0,7))
    S2e = df_test['S2e']
    result_med_fun = get_median(df_test)
    result_med_fun = result_med_fun['mu']
    result_med_data = 8100

    assert (result_med_fun == result_med_data).all()




def test_get_median_values():
    S2e_test1 = pd.DataFrame([1, 2, 3],
                             columns = ['S2e'])

    S2e_test2 = pd.DataFrame([1, 1, 1, 1,
                              2, 2, 2, 2,
                              3, 3, 3, 3],
                             columns = ['S2e'])

    map_test1  = get_median(S2e_test1)
    map_test2  = get_median(S2e_test2)

    ratio_error_values  = map_test1['mu_error'].values / map_test2['mu_error'].values
    ratio_errors = ((map_test1['sigma']/np.sqrt(len(S2e_test1)))/(map_test2['sigma']/np.sqrt(len(S2e_test2)))).values

    assert map_test1['mu'].values == map_test2['mu'].values
    assert map_test1['sigma'].values == map_test2['sigma'].values
    assert ratio_error_values == ratio_errors




def test_gaussian_fit_few_entries():
    empty_dst = pd.DataFrame(columns = ['DT', 'x', 'y', 'S2e'])
    result_fit = gaussian_fit(empty_dst, 1, min_events = 10)
    result_fun = get_median(empty_dst)['mu']

    assert np.allclose(result_fit.mu.values, result_fun.values, equal_nan=True)




def test_gaussian_fit_get_median_Nevents():
    d = {'DT': np.empty(6), 'x' : np.empty(6), 'y' : np.empty(6),'S2e' : np.ones(6)}
    df_test = pd.DataFrame(data = d, index = range(0, 6))
    N = len(df_test)
    result_med = get_median(df_test)
    result_fit = gaussian_fit(df_test, ebins = 6)
    assert result_med['nevents'].iloc[0] == N
    assert result_fit['nevents'].iloc[0] == N





def test_gaussianfit_computes_right_values():

    with fix_random_seed(42):
        S2e_df = pd.DataFrame({
            'S2e': np.random.normal(loc = 8000, scale = 10.0, size = 10000)
        })

    results = gaussian_fit(S2e_df, ebins = 50)

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
                          bins = 25) #ebins in gaussian_fit()

    reference = result.iloc[0]

    for _, row in result.iterrows():
        assert np.allclose(result.mu , reference.mu)
        assert np.allclose(row['sigma'], reference['sigma'])
        assert np.isclose(row['mu_error'], reference['mu_error'])
        assert np.isclose(row['sigma_error'], reference['sigma_error'])
        assert row['nevents'] == reference['nevents']





def test_merge_maps():
    Nan_map_test = create_empty_map(xy_range = (-500, 500), dt_range = (0, 1400), xy_nbins = 100, dt_nbins = 10)

    d = {'k':np.array([1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3]),
         'i':np.array([1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 2]),
         'j':np.array([1, 2, 1, 2, 1, 1, 2, 1, 2, 1, 2]),
         'nevents': np.empty(11, dtype = int),
         'mu' : np.empty(11),
         'sigma' : np.empty(11),
         'mu_error' : np.empty(11),
         'sigma_error' : np.empty(11)} #pon 2 is, 2js, 3ks por exemplo

    map_3D_test = pd.DataFrame(data = d, index = range(0, 11))
    assert merge_maps(Nan_map_test, map_3D_test).shape == Nan_map_test.shape



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

