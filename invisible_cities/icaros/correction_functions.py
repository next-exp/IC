import pandas as pd
import numpy  as np
from scipy.interpolate import griddata
from invisible_cities.core.core_functions import in_range


def normalization(krmap, method, xy_params = None):

    krmap = krmap.dropna(subset=['mu'])

    anode = krmap[krmap.k == 0]

    if method == 'max':
        E_reference_max = krmap.mu.max()
        return E_reference_max

    if method == 'mean chamber':
        E_reference_chamber = krmap.mu.mean()
        return E_reference_chamber

    if method == 'median chamber':
        E_median_chamber = krmap.mu.median()
        return E_median_chamber

    if method == 'mean anode':
        E_reference_anode = anode.mu.mean()
        return E_reference_anode

    if method == 'median anode':
        E_median_anode = anode.mu.median()
        return E_median_anode

    mask_region = ( in_range(krmap.x, xy_params['x_low'], xy_params['x_high']) &
                    in_range(krmap.y, xy_params['y_low'], xy_params['y_high']) )
    krmap = krmap[mask_region]

    if method == 'mean region chamber':
        region = krmap[mask_region]
        E_reference_region = region.mu.mean()
        return E_reference_region

    if method == 'median region':
        region = krmap[mask_region]
        E_median_region = region.mu.median()
        return E_median_region

    mask_region_anode = mask_region & (krmap.k == 0)
    anode = krmap[mask_region_anode]

    if method == 'mean region anode':
        E_reference_slice_anode = anode.mu.mean()
        return E_reference_slice_anode

    if method == 'median region anode':
        E_median_region_anode = anode.mu.median()
        return E_median_region_anode




def apply_3Dmap(krmap, norm_method, dt, x, y, E, xy_params = None, keV = False):

    map_points = krmap['dt x y'.split()].values
    norm = normalization(krmap, norm_method, xy_params)

    data_points = np.stack([dt, x, y], axis = 1)
    E_interpolated_data = griddata(map_points, krmap.mu.values, data_points, method = 'nearest')

    correction_factor = norm/E_interpolated_data
    Ec = E * correction_factor

    if keV:
        Ec = Ec * (41.55 / norm)

    return Ec
