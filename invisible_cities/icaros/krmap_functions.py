import pandas as pd
import numpy  as np

from invisible_cities.core.fit_functions import expo, gauss, fit
from invisible_cities.io.dst_io import load_dst, load_dsts, df_writer
from invisible_cities.core.core_functions import in_range, shift_to_bin_centers
from invisible_cities.core.stat_functions import poisson_sigma

from lifetime_vdrift_functions import LT_fit, select_lifetime_region, compute_drift_v

import tables            as tb
from   scipy             import stats
from   scipy             import optimize
from   scipy.optimize import curve_fit

import itertools

"""

This script contains functions to compute empty maps, 3D maps from kdsts and merge them to create an 'uniform' map.
It also contains functions to get and store map info (metadata) and time evolution.

"""

def create_empty_map(xy_range, dt_range, xy_nbins, dt_nbins):
    """
    - Uses every possible indices combination
    - Creates a DataFrame filled with NaNs for each (i, j, k) combination
    """

    i_range = np.arange(0, xy_nbins)
    j_range = np.arange(0, xy_nbins)
    k_range = np.arange(0, dt_nbins)

    combinations = pd.DataFrame(itertools.product(k_range, i_range, j_range),
                                columns=['k', 'i', 'j'])
    Nan_map = pd.DataFrame(
        np.nan,
        index=range(len(combinations)),
        columns=['nevents', 'mu', 'sigma', 'mu_error', 'sigma_error']
    )
    Nan_map = pd.concat([combinations, Nan_map], axis=1)

    return Nan_map


def get_median(df):
    sigma = (0.04/2.35)*df.S2e.median()
    #Kr energy resolution is ~4%, so the std for S2e median would be ~0.04/2.35

    return pd.DataFrame({
         'nevents': len(df),
         'median': df.S2e.median(),
         'sigma': sigma,
         'median_error': sigma/np.sqrt(len(df)),
         'sigma_error': np.nan}, index = [0]) #error de std?



def gaussian_fit(df, ebins, min_events = 10):
    if len(df)< min_events:
        results = get_median(df)

    counts, bin_edges = np.histogram(df.S2e, ebins)
    bin_centers = shift_to_bin_centers(bin_edges)

    try:
        f = fit(gauss, bin_centers, counts, seed = (len(df), 8000, 400), maxfev = 2500)
    except:

        results = pd.DataFrame({'nevents': len(df), 'mu': np.nan, 'sigma': np.nan, 'mu_error': np.nan, 'sigma_error': np.nan}, index = [0])

        return results
    results = pd.DataFrame({'nevents': len(df), 'mu': f.values[1], 'sigma': f.values[2], 'mu_error': f.errors[1], 'sigma_error': f.errors[2]}, index = [0])

    return results


def map_3D_fits(df, xy_range, dt_range, xy_nbins, dt_nbins):

    df = df.loc[:,['X', 'Y', 'DT', 'S2e']]

    mask = in_range(df.X, *xy_range) & in_range(df.Y, *xy_range) & in_range(df.DT, *dt_range)

    df = df[mask]

    x = df.X
    y = df.Y
    dt = df.DT

    xy_bins = np.linspace(*xy_range, xy_nbins + 1)
    dt_bins = np.linspace(*dt_range, dt_nbins + 1)

    map_df = df.assign( i = np.digitize(x.values, bins = xy_bins) - 1,
                        j = np.digitize(y.values, bins = xy_bins) - 1,
                        k = np.digitize(dt.values, bins = dt_bins) - 1)


    result = map_df.groupby(['k', 'i', 'j']).apply(fit_function, bins = np.linspace(3000, 9000, 101))

    result = result.reset_index()

    if 'level_3' in result.columns:
        result = result.drop(columns = 'level_3')

    return result


def merge_maps(NaN_map, map_3D):

    full_map = pd.concat([map_3D, NaN_map], ignore_index = True)
    full_map = full_map.groupby(['k', 'i','j']).first().reset_index()

    return full_map


def include_coordinates(map, xy_range, dt_range, xy_nbins, dt_nbins):

    xy_bins = np.linspace(*xy_range, xy_nbins + 1)
    dt_bins = np.linspace(*dt_range, dt_nbins + 1)

    x = shift_to_bin_centers(xy_bins)
    y = shift_to_bin_centers(xy_bins)
    dt = shift_to_bin_centers(dt_bins)

    combinaciones_bins = pd.DataFrame(
    itertools.product(dt, x, y),
    columns=['dt', 'x', 'y'])


    map = pd.concat([combinaciones_bins.reset_index(drop=True), map.reset_index(drop=True)], axis = 1)

    return map


def compute_3D_map(df, xy_range, dt_range, xy_nbins, dt_nbins):
    NaN_map = create_NaN_map(xy_range, dt_range, xy_nbins, dt_nbins)
    map_3D_fit = map_3D_fits(df, xy_range = xy_range, dt_range = dt_range, xy_nbins = xy_nbins, dt_nbins = dt_nbins)

    map = merge_maps(NaN_map, map_3D_fit)
    full_map = include_coordinates(map, xy_range, dt_range, xy_nbins, dt_nbins)

    return full_map

#def compute_metadata():
#    ['rmax', 'zmax', 'bin_size_z', 'bin_size_x', 'bin_size_y', 'method',
#       'zbins', 'xbins', 'ybins', 'nbins_z', 'nbins_x', 'nbins_y', 'map_shape',
#       'map_extent']

#def save_map(map, metadata)
