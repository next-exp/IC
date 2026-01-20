import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from correction_functions import normalization, apply_3Dmap


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

    region_mask = (
        (map_test.x <= 100) & (map_test.x >= -100) &
        (map_test.y <= 100) & (map_test.y >= -100)
    )

    assert normalization(map_test, 'max', None, None, None, None) == 9889
    assert normalization(map_test, 'mean chamber', None, None, None, None) == 4536.125
    assert normalization(map_test, 'mean anode', None, None, None, None) == 4400
    assert normalization(map_test, 'mean region anode', -100, 100, -100, 100) == map_test.loc[(map_test.k == 0) & region_mask,'mu'].mean()
    assert normalization(map_test, 'mean region chamber', -100, 100, -100, 100) == map_test.loc[region_mask,'mu'].mean()
    
    assert normalization(map_test, 'median chamber', None, None, None, None) == 4647.5
    assert normalization(map_test, 'median anode', None, None, None, None) == 4400.0
    assert normalization(map_test, 'median region anode', -100, 100, -100, 100) == map_test.loc[(map_test.k == 0) & region_mask, 'mu'].median()
    assert normalization(map_test, 'median region chamber', -100, 100, -100, 100) == map_test.loc[region_mask, 'mu'].median()
    
    
def test_apply_3Dmap():
    
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
    
    map_points = map_test['dt x y'.split()].values
    
    E_interpolated_data = griddata(map_points, map_test.mu.values, data_points_test, method = 'nearest')
    
    Ec = apply_3Dmap(map_test, norm_method = 'max', dt, x, y, E, keV = True)

    assert len(Ec) == len(dt)
    assert len(E_interpolated_data) == len(data_points_test)
    
