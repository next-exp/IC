import pandas as pd
import numpy  as np
from scipy.interpolate import griddata



def normalization(krmap, method, x_low, x_high, y_low, y_high):

    mu_values = krmap.mu

    if method == 'max':
        E_reference_max = mu_values.dropna().max()
        return E_reference_max

    if method == 'mean chamber':
        E_reference_chamber = mu_values.dropna().mean()
        return E_reference_chamber

    if method == 'mean anode':
        mu_values_anode = map[map.k == 0].mu
        E_reference_anode = mu_values_anode.dropna().mean()
        return E_reference_anode

    mask_region = (map['x'] <= x_high) & (map['x'] >= x_low) & (map['y'] <= y_high) & (map['y'] >= y_low)

    if method == 'mean region anode':
        region = map[map.k == 0][mask_region]
        E_reference_slice_anode = region.mu.dropna().mean()
        return E_reference_slice_anode

    if method == 'mean region chamber':
        region = map[mask_region]
        E_reference_region = region.mu.dropna().mean()
        return E_reference_region


def apply_3Dmap(krmap, method, dt, x, y, E, keV = False):

    map_points = krmap['dt x y'.split()].values
    norm = normalization(krmap, method)

    data_points = np.stack([dt, x, y], axis = 1)
    E_interpolated_data = griddata(map_points, krmap.mu.values, data_points, method = 'nearest')

    correction_factor = norm/E_interpolated_data
    Ec = E * correction_factor

    if keV:
        Ec = Ec * (41.55 / norm)

    return Ec
