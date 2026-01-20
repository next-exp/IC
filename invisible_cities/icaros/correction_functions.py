import pandas as pd
import numpy  as np
from scipy.interpolate import griddata


def normalization(krmap, method, xy_params = None):

    mu_values = krmap.mu.dropna()
    
    anode = krmap[krmap.k == 0]

    if method == 'max':
        E_reference_max = mu_values.max()
        return E_reference_max

    if method == 'mean chamber':
        E_reference_chamber = mu_values.mean()
        return E_reference_chamber
    
    if method == 'median chamber':
        E_median_chamber = mu_values.median()
        return E_median_chamber
    
    if method == 'mean anode':
        mu_values_anode = anode.mu
        E_reference_anode = mu_values_anode.mean()
        return E_reference_anode
    
    if method == 'median anode':
        mu_median_anode = anode.mu
        E_median_anode = mu_median_anode.median()
        return E_median_anode
        
    mask_region = (krmap['x'] <= xy_params['x_high']) & (krmap['x'] >= xy_params['x_low']) & (krmap['y'] <= xy_params['y_high']) & (krmap['y'] >= xy_params['y_low'])

    if method == 'mean region anode':
        region = anode[mask_region]
        E_reference_slice_anode = region.mu.mean()
        return E_reference_slice_anode

    if method == 'mean region chamber':
        region = krmap[mask_region]
        E_reference_region = region.mu.mean()
        return E_reference_region
    
    if method == 'median region anode':
        region = anode[mask_region]
        E_median_region_anode = region.mu.median()
        return E_median_region_anode
    
    if method == 'median region':
        region = krmap[mask_region]
        E_median_region = region.mu.median()
        return E_median_region


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
        
        
    
