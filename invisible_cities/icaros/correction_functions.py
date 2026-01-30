import pandas as pd
import numpy  as np
from scipy.interpolate import griddata
from invisible_cities.core.core_functions import in_range
from invisible_cities.types.symbols import NormMethod


def normalization(krmap, method, xy_params = None):

    krmap = krmap.dropna(subset=['mu'])

    anode = krmap[krmap.k == 0]

    if method is NormMethod.max:
        E_reference_max = krmap.mu.max()
        return E_reference_max

    if method is NormMethod.mean_chamber:
        E_reference_chamber = krmap.mu.mean()
        return E_reference_chamber

    if method is NormMethod.median_chamber:
        E_median_chamber = krmap.mu.median()
        return E_median_chamber

    if method is NormMethod.mean_anode:
        E_reference_anode = anode.mu.mean()
        return E_reference_anode

    if method is NormMethod.median_anode:
        E_median_anode = anode.mu.median()
        return E_median_anode

    mask_region = ( in_range(krmap.x, xy_params['x_low'], xy_params['x_high']) &
                    in_range(krmap.y, xy_params['y_low'], xy_params['y_high'])
                   ).values

    krmap = krmap[mask_region]

    if method is NormMethod.mean_region_chamber:
        E_reference_region = krmap.mu.mean()
        return E_reference_region

    if method is NormMethod.median_region_chamber:
        E_median_region = krmap.mu.median()
        return E_median_region

    anode = krmap[krmap.k == 0]

    if method is NormMethod.mean_region_anode:
        E_reference_slice_anode = anode.mu.mean()
        return E_reference_slice_anode

    if method is NormMethod.median_region_anode:
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


def apply_correctionmap(kdst, map3D, norm_method, xy_params, col_name, keV = True):

    """
    xy_params must be a dictionary: {'x_high': , 'x_low': , 'y_high': , 'y_low': }

    """

    corrected_energy = apply_3Dmap(map3D, norm_method, kdst.DT, kdst.X, kdst.Y, kdst.S2e, xy_params = xy_params, keV = keV)

    kdst[col_name] = corrected_energy.values


    return kdst
