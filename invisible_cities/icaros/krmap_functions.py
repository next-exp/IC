import pandas as pd
import numpy  as np

from invisible_cities.core.fit_functions import expo, gauss, fit
from invisible_cities.io.dst_io import load_dst, load_dsts, df_writer
from invisible_cities.core.core_functions import in_range, shift_to_bin_centers, weighted_mean_and_std
from invisible_cities.core.stat_functions import poisson_sigma

from lifetime_vdrift_functions import  select_lifetime_region, compute_drift_v

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
         'mu': df.S2e.median(),
         'sigma': sigma,
         'mu_error': sigma/np.sqrt(len(df)),
         'sigma_error': np.nan}, index = [0]) #error de std?



def gaussian_fit(df, ebins, min_events = 10):
    if len(df)< min_events:
        return get_median(df)

    counts, bin_edges = np.histogram(df.S2e, ebins)
    bin_centers = shift_to_bin_centers(bin_edges)

    try:
        f = fit(gauss, bin_centers, counts, seed = (len(df), 8000, 400), maxfev = 2500)
    except:

        results = pd.DataFrame({'nevents': len(df), 'mu': np.nan, 'sigma': np.nan, 'mu_error': np.nan, 'sigma_error': np.nan}, index = [0])

        return results
    results = pd.DataFrame({'nevents': len(df), 'mu': f.values[1], 'sigma': f.values[2], 'mu_error': f.errors[1], 'sigma_error': f.errors[2]}, index = [0])

    return results


def fit_map(df, xy_range, dt_range, xy_nbins, dt_nbins, fit_function, bins):

    df = df.loc[:,['X', 'Y', 'DT', 'S2e']]

    mask = in_range(df.X, *xy_range) & in_range(df.Y, *xy_range) & in_range(df.DT, *dt_range)

    df = df[mask]

    x, y, dt, _ = df.values.T

    xy_bins = np.linspace(*xy_range, xy_nbins + 1)
    dt_bins = np.linspace(*dt_range, dt_nbins + 1)

    map_df = df.assign( i = np.digitize(x, bins = xy_bins) - 1,
                        j = np.digitize(y, bins = xy_bins) - 1,
                        k = np.digitize(dt, bins = dt_bins) - 1)

    #add -1 so the indices go from 0 to 49, not from 1 to 50.

    result = map_df.groupby(['k', 'i', 'j']).apply(fit_function, bins)

    result = result.reset_index()
    result = result.drop(columns = 'level_3')

    return result


def merge_maps(NaN_map, map_3D):

    """
    Merge empty and 3D (data) maps, for those "repeated rows" with NaNs and data,
    we discard the NaN rows (doing .first()) to get a dataframe with "uniform" shape.
    """

    full_map = pd.concat([map_3D, NaN_map], ignore_index = True)
    full_map = full_map.groupby(['k', 'i','j']).first().reset_index()

    return full_map


def include_coordinates(krmap, xy_range, dt_range, xy_nbins, dt_nbins):

    xy_bins = np.linspace(*xy_range, xy_nbins + 1)
    dt_bins = np.linspace(*dt_range, dt_nbins + 1)

    x = shift_to_bin_centers(xy_bins)
    y = shift_to_bin_centers(xy_bins)
    dt = shift_to_bin_centers(dt_bins)

    bin_combinations = pd.DataFrame(
    itertools.product(dt, x, y),
    columns=['dt', 'x', 'y'])


    krmap = pd.concat([bin_combinations, krmap], axis = 1)

    return krmap


def compute_3D_map(df, xy_range, dt_range, xy_nbins, dt_nbins, fit_function, bins):
    NaN_map = create_empty_map(xy_range, dt_range, xy_nbins, dt_nbins)
    map_3D_fit = fit_map(df, xy_range, dt_range, xy_nbins, dt_nbins, fit_function, bins)

    map = merge_maps(NaN_map, map_3D_fit)
    full_map = include_coordinates(map, xy_range, dt_range, xy_nbins, dt_nbins)

    return full_map



def compute_metadata(df, krmap, xy_range, dt_range,
                     xy_nbins, dt_nbins, norm_method):

    metadata = {
        'rmax'        : df.R.max(),
        'zmax'        : df.Z.max(),
        'bin_size_dt' : (dt_range[1] - dt_range[0]) / dt_nbins,
        'bin_size_x'  : (xy_range[1] - xy_range[0]) / xy_nbins,
        'bin_size_y'  : (xy_range[1] - xy_range[0]) / xy_nbins,
        'method'      : norm_method,
        'dtbins'      : [np.unique(krmap.k.values).tolist()],
        'xbins'       : [np.unique(krmap.i.values).tolist()],
        'ybins'       : [np.unique(krmap.j.values).tolist()],
        'nbins_dt'    : dt_nbins,
        'nbins_x'     : xy_nbins,
        'nbins_y'     : xy_nbins,
        'map_shape'   : [df.shape],
        'map_extent'  : len(df),
    }

    #metadata_str = {k: str(v) for k, v in metadata.items()}
    #I had to add this line so df_writer doesnt rise an error

    return pd.DataFrame(metadata, index = [0]).T



def gauss_seed(x, y, sigma_rel=0.05):
    """
    Estimate the seed for a gaussian fit to the input data.
    """
    y_max  = np.argmax(y) # highest bin
    x_max  = x[y_max]
    sigma  = sigma_rel * x_max
    amp    = y_max * (2 * np.pi)**0.5 * sigma * np.diff(x)[0]
    seed   = amp, x_max, sigma


    return seed


def quick_gauss_fit(data, bins, sigma = False):
    """
    Histogram input data and fit it to a gaussian with the parameters
    automatically estimated.
    """
    y, x  = np.histogram(data, bins)
    x     = shift_to_bin_centers(x)
    seed  = gauss_seed(x, y)

    if sigma:
        sigma = poisson_sigma(y)
        f     = fit(gauss, x, y, seed, sigma = sigma)
    else:
        f     = fit(gauss, x, y, seed)
    assert np.all(f.values != seed)
    return f



def create_time_slices(df, run_number, slice_hours):
    slice_seconds = slice_hours * 3600

    t_min = df.time.min()
    t_max = df.time.max()

    time_edges = np.arange(t_min, t_max + slice_seconds, slice_seconds)

    dataframes = []

    for i in range(len(time_edges) - 1):
        t_start = time_edges[i]
        t_end   = time_edges[i + 1]

        mask = (df.time >= t_start) & (df.time < t_end)
        df_slice = df[mask]
        dataframes.append(df_slice)

    return dataframes



def get_time_evol(df, run_number, bins_lt, x0, y0, shape, shape_size, p0, dtrange, low_DT, high_DT, bins_Ec, error = False, seed = None):

    df_tevols = []

    for i in range(len(df)):

        mean = df[i].S2e.mean()
        std = df[i].S2e.std()/np.sqrt(len(df[i]))

        kdst_in_region = select_lifetime_region(df[i], bins_lt, x0, y0, shape, shape_size)
        magnitudes, uncertainties = LT_fit(kdst_in_region.DT, kdst_in_region.S2e, p0)
        lifetime = -magnitudes[1]
        ulifetime = uncertainties[1]

        e0 = magnitudes[0]
        e0u = uncertainties[0]

        dv, udv = compute_drift_v(df[i].DT.to_numpy(), df[i].DT.nunique(), dtrange, seed)

        f = quick_gauss_fit(df[i].Ec_2.values, bins_Ec, sigma = error)

        mu = f.values[1]
        u_mu = f.errors[1]

        sigma = f.values[2]
        u_sigma = f.errors[2]

        R = np.sqrt(8*np.log(2))*(sigma/mu) #FWHM (krypton paper)
        u_R = R * (u_mu**2/mu**2 + u_sigma**2/sigma**2)**0.5

        mask = (df[i].DT.values > low_DT) & (df[i].DT.values < high_DT)
        s1e_cath = df[i][mask].S1e.values.mean()
        us1e_cath = df[i][mask].S1e.values.std()/np.sqrt(len(df[i]))

        t_evol = {'run_number' : run_number,
              'ts' : int(df[i].time.values[0]),
              's2e': mean,
              's2eu' : std,
              'ec' : mu,
              'ecu': u_mu,
              'chi2_ec': f.chi2,
              'e0': e0,
              'e0u': e0u,
              'lt' : lifetime ,
              'ltu' : ulifetime,
              'dv' : dv,
              'dvu' : udv,
              'resol' : R*100,
              'resolu' : u_R*100,
              's1w' : df[i].S1w.mean(),
              's1wu' : df[i].S1w.std()/np.sqrt(len(df[i])),
              's1e' : s1e_cath,
              's1eu' : us1e_cath,
              's1h': df[i].S1h.mean(),
              's1hu':df[i].S1h.std()/np.sqrt(len(df[i])),
              's2w':df[i].S2w.mean(),
              's2wu': df[i].S2w.std()/np.sqrt(len(df[i])),
              's2q': df[i].S2q.mean(),
              's2qu': df[i].S2q.std()/np.sqrt(len(df[i])),
              'Nsipm': df[i].Nsipm.mean(),
              'Nsipmu': df[i].Nsipm.std()/np.sqrt(len(df[i])),
              'Xrms': df[i].Xrms.mean(),
              'Xrmsu': df[i].Xrms.std()/np.sqrt(len(df[i])),
              'Yrms': df[i].Yrms.mean(),
              'Yrmsu': df[i].Yrms.std()/np.sqrt(len(df[i])),
              'Zrms': df[i].Zrms.mean(),
              'Zrmsu': df[i].Zrms.std()/np.sqrt(len(df[i])),
               }

        df_tevol = pd.DataFrame(data = t_evol, index = [0])

        df_tevols.append(df_tevol)


    return pd.concat(df_tevols, ignore_index = True)



def save_map(name, efficiencies, krmap, metadata, t_evol):
    metadata.to_hdf(name, key = 'metadata', mode = 'w')

    with tb.open_file(name, "a") as file:
        df_writer(file, efficiencies, group_name = 'data', table_name = 'selection_efficiencies')
        df_writer(file, krmap, group_name = 'krmap', table_name = 'krmap')
        #df_writer(file, metadata, group_name = 'krmap_info', table_name = 'metadata')
        df_writer(file, t_evol, group_name = 't_evol', table_name = 't_evol')
