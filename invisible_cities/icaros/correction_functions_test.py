import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from correction_functions import normalization, apply_3Dmap, apply_correctionmap
from invisible_cities.core.core_functions import in_range
from invisible_cities.types.symbols import NormMethod
from scipy.interpolate import griddata


def test_method_norm():

    k_vals = np.arange(10)
    i_vals = np.arange(12)
    j_vals = np.arange(12)


    k, i, j = map(np.ravel,
                  np.meshgrid(k_vals, i_vals, j_vals, indexing= 'ij'))

    x = i * 25
    y = j * 25
    dt = k * 45

    mu = (k*i + 800)*j

    map_test = pd.DataFrame({
        'k': k,
        'i': i,
        'j': j,
        'x': x,
        'y': y,
        'dt': dt,
        'mu': mu
    })

    region = {'x_low': -100,
              'x_high': 100,
              'y_low': -100,
              'y_high': 100}


    region_mask = (
        in_range(map_test.x, region['x_low'],region['x_high']) &
        in_range(map_test.y, region['y_low'],region['y_high'])
    )


    assert normalization(map_test, NormMethod.max) == 9889
    assert normalization(map_test, NormMethod.mean_chamber) == 4536.125
    assert normalization(map_test, NormMethod.mean_anode) == 4400
    assert normalization(map_test, NormMethod.median_chamber) == 4647.5
    assert normalization(map_test, NormMethod.median_anode) == 4400.0

    assert normalization(map_test, NormMethod.mean_region_anode, region) == map_test.loc[(map_test.k == 0) & region_mask,'mu'].mean()
    assert normalization(map_test, NormMethod.mean_region_chamber, region) == map_test.loc[region_mask,'mu'].mean()
    assert normalization(map_test, NormMethod.median_region_anode, region) == map_test.loc[(map_test.k == 0) & region_mask, 'mu'].median()
    assert normalization(map_test, NormMethod.median_region_chamber, region) == map_test.loc[region_mask, 'mu'].median()




def test_apply_3Dmap():

    k_vals = np.arange(10)+1
    i_vals = np.arange(12)+1
    j_vals = np.arange(12)+1


    k, i, j = map(np.ravel,
                  np.meshgrid(k_vals, i_vals, j_vals, indexing= 'ij'))

    x = i * 25
    y = j * 25
    dt = k * 45

    mu = (k*i + 800)*j

    map_test = pd.DataFrame({
        'k': k,
        'i': i,
        'j': j,
        'x': x,
        'y': y,
        'dt': dt,
        'mu': mu
    })


    n = len(map_test)
    dt = np.linspace(20, 1350, n)
    x  = np.linspace(-450, 450, n)
    y  = x
    E = np.linspace(7e3, 9e3, n)
    data_points_test = np.stack([dt, x, y], axis = 1)

    Ec = apply_3Dmap(map_test, NormMethod.max, dt, x, y, E, keV = True)

    assert len(Ec) == len(dt)



def test_apply_3Dmap_same_values():
    n = 100
    dt = np.linspace(20, 1350, n)
    x  = np.linspace(-450, 450, n)
    y  = x
    E = np.linspace(7e3, 9e3, n)
    data_points_test = np.stack([dt, x, y], axis = 1)


    k_vals = np.arange(10)
    i_vals = np.arange(12)
    j_vals = np.arange(12)


    k, i, j = map(np.ravel,
                  np.meshgrid(k_vals, i_vals, j_vals, indexing= 'ij'))

    x = i * 25
    y = j * 25
    dt = k * 45

    mu = 800

    map_test = pd.DataFrame({
        'k': k,
        'i': i,
        'j': j,
        'x': x,
        'y': y,
        'dt': dt,
        'mu': mu
    })


    map_points = map_test['dt x y'.split()].values
    norm = normalization(map_test,  NormMethod.max, xy_params = None)

    data_points = np.stack([dt, x, y], axis = 1)
    E_interpolated_data = griddata(map_points, map_test.mu.values, data_points, method = 'nearest')
    correction_factor = norm/E_interpolated_data


    map_test.loc[0, 'mu'] = 1600
    map_points2 = map_test['dt x y'.split()].values
    norm2 = normalization(map_test, NormMethod.max, xy_params = None)

    E_interpolated_data2 = griddata(map_points2, map_test.mu.values, data_points, method = 'nearest')

    correction_factor2 = norm2/E_interpolated_data2

    assert (correction_factor == 1).all()
    assert (correction_factor2[1:] == 2).all()



def test_apply_correctionmap_shape():
    x = np.linspace(0, 10, 11)

    kdst = pd.DataFrame({'X':x,
                         'Y':-x,
                         'DT': np.linspace(20, 1000, 11),
                         'S2e': np.linspace(7500, 8500, 11)
                         })

    kdst_test = kdst.copy()

    map_3D = pd.DataFrame({'k':np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2]),
                           'i':np.array([0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1]),
                           'j':np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1]),
                           'nevents': np.linspace(10000, 12000, 11, dtype=int),
                           'mu' : np.linspace(7900, 8100, 11),
                           'sigma' : np.linspace(10, 50, 11),
                           'mu_error' : np.linspace(0, 10, 11),
                           'sigma_error' : np.linspace(0, 2, 11),
                           'dt': np.linspace(20, 1000, 11),
                           'x': x,
                           'y': -x
                           })

    kdst_correct = apply_correctionmap(kdst, map_3D, norm_method = NormMethod.max, xy_params = None, col_name='Ec')

    assert kdst_correct.shape[1] ==  kdst_test.shape[1] + 1
    assert ((kdst_test == kdst_correct.drop(columns = 'Ec')).all()).all()
