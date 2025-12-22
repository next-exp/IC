import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from correction_functions import normalization


def test_method_norm():

    k_vals = np.arange(10)
    i_vals = np.arange(12)
    j_vals = np.arange(12)

    k, i, j = np.meshgrid(k_vals, i_vals, j_vals, indexing='ij')
    k = k.ravel()
    i = i.ravel()
    j = j.ravel()

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

    assert normalization(map_test, 'max', -100, 100, -100, 100) == 9889
    assert normalization(map_test, 'mean chamber', -100, 100, -100, 100) == 4536.125
    assert normalization(map_test, 'mean anode', -100, 100, -100, 100) == 4400
    assert normalization(map_test, 'mean region anode', -100, 100, -100, 100) == map_test.loc[(map_test.k == 0) & region_mask,'mu'].mean()
    assert normalization(map_test, 'mean region chamber', -100, 100, -100, 100) == map_test.loc[region_mask,'mu'].mean()
