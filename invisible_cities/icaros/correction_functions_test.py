import numpy as np
import pandas as pd
from invisible_cities.icaros.correction_functions import normalization, apply_3Dmap, apply_correctionmap_inplace_kdst
from invisible_cities.core.core_functions import in_range
from invisible_cities.types.symbols import NormMethod
from scipy.interpolate import griddata
from pytest import fixture


@fixture
def dummy_map():
    k_vals = np.arange(10)
    i_vals = np.arange(12)
    j_vals = np.arange(12)


    k, i, j = map(np.ravel,
                  np.meshgrid(k_vals, i_vals, j_vals, indexing= 'ij'))

    x = i * 25
    y = j * 25
    dt = k * 45

    mu = (k*i + 800)*j + 1

    map_test = pd.DataFrame({
        'k': k,
        'i': i,
        'j': j,
        'x': x,
        'y': y,
        'dt': dt,
        'mu': mu
    })

    return map_test


def test_normalization(dummy_map):

    dummy_map['mu'] = dummy_map.mu - 1

    region = {'x_low': -100,
              'x_high': 100,
              'y_low': -100,
              'y_high': 100}


    region_mask = (
        in_range(dummy_map.x, region['x_low'],region['x_high']) &
        in_range(dummy_map.y, region['y_low'],region['y_high'])
    )


    assert normalization(dummy_map, NormMethod.maximum) == 9889
    assert normalization(dummy_map, NormMethod.mean_chamber) == 4536.125
    assert normalization(dummy_map, NormMethod.mean_anode) == 4400
    assert normalization(dummy_map, NormMethod.median_chamber) == 4647.5
    assert normalization(dummy_map, NormMethod.median_anode) == 4400.0

    assert normalization(dummy_map, NormMethod.mean_region_anode, region) == 1200.0
    assert normalization(dummy_map, NormMethod.mean_region_chamber, region) == 1210.125
    assert normalization(dummy_map, NormMethod.median_region_anode, region) == 1200.0
    assert normalization(dummy_map, NormMethod.median_region_chamber, region) == 1213.5




def test_apply_3Dmap(dummy_map):

    Ec = apply_3Dmap(dummy_map, NormMethod.maximum, dummy_map.dt, dummy_map.x, dummy_map.y, dummy_map.mu, keV = True)

    assert len(Ec) == len(dummy_map)



def test_apply_3Dmap_same_values(dummy_map):

    dummy_map['mu'] = 8000

    Ec = apply_3Dmap(dummy_map, norm_method = NormMethod.maximum, dt = dummy_map.dt, x = dummy_map.x, y = dummy_map.y, E = dummy_map.mu, keV = False)

    assert (Ec == dummy_map.mu).all()

def test_apply_3Dmap_one_different_value(dummy_map):

    dummy_map['mu'] = 8000
    dummy_map.loc[0, 'mu'] = 2*8000

    Ec = apply_3Dmap(dummy_map, norm_method = NormMethod.maximum, dt =dummy_map.dt, x = dummy_map.x, y = dummy_map.y, E = dummy_map.mu, keV = False)

    assert (Ec[1:] == 2*dummy_map.mu[1:]).all()



def test_apply_correctionmap_shape(dummy_map):
    dummy = np.linspace(0, 10, 11)

    kdst = pd.DataFrame({'X':dummy,
                         'Y':-dummy,
                         'DT': dummy,
                         'S2e': dummy
                         })

    kdst_test = kdst.copy()

    kdst_correct = apply_correctionmap_inplace_kdst(kdst, dummy_map, norm_method = NormMethod.maximum, xy_params = None, col_name='Ec')

    assert kdst_correct.shape[1] ==  kdst_test.shape[1] + 1
    assert ((kdst_test == kdst_correct.drop(columns = 'Ec')).all()).all()
