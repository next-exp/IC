import pandas as pd
import numpy  as np

from invisible_cities.core.fit_functions import expo, gauss, fit
from invisible_cities.io.dst_io import load_dst, load_dsts, df_writer
from invisible_cities.core.core_functions import in_range, shift_to_bin_centers, weighted_mean_and_std
from invisible_cities.core.stat_functions import poisson_sigma

from lifetime_vdrift_functions import select_lifetime_region, compute_drift_v

import tables            as tb
from   scipy             import stats
from   scipy             import optimize
from   scipy.optimize import curve_fit

import itertools

"""

This script contains functions to compute empty maps, 3D maps from kdsts and merge them to create an 'uniform' map.
It also contains functions to get and store map info (metadata) and time evolution.

"""


def gauss_seed(x, y, sigma_rel=0.05):
    """
    Estimate the seed for a gaussian fit to the input data.
    """
    x_max, sigma = weighted_mean_and_std(x, y)
    amp    = y.max()  * (2 * np.pi)**0.5 * sigma
    seed   = amp, x_max, sigma

    return seed



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
    '''
    We consider that for len(df) < 5, it makes no sense to compute the std

    Kr energy resolution is ~4%, so the std for S2e median would be ~0.04/2.35
    '''

    if len(df) < 5:
        sigma = (0.04/2.35)*df.S2e.median()
    else:
        sigma = df.S2e.std()

    return pd.DataFrame({
         'nevents': len(df),
         'mu': df.S2e.median(),
         'sigma': sigma,
         'mu_error': sigma/np.sqrt(len(df)),
         'sigma_error': np.nan}, index = [0])



def gaussian_fit(df, nbins_S2e, min_events = 10):
    if len(df)< min_events:
        return get_median(df)

    counts, bin_edges = np.histogram(df.S2e, nbins_S2e)
    bin_centers = shift_to_bin_centers(bin_edges)

    seed = gauss_seed(bin_centers, counts)

    try:
        f = fit(gauss, bin_centers, counts, seed = seed, maxfev = 2500)
    except:


        return pd.DataFrame({'nevents': len(df),
                                 'mu': np.nan,
                                 'sigma': np.nan,
                                 'mu_error': np.nan,
                                 'sigma_error': np.nan}, index = [0])


    return  pd.DataFrame({'nevents': len(df),
                            'mu': f.values[1],
                            'sigma': f.values[2],
                            'mu_error': f.errors[1],
                            'sigma_error': f.errors[2]}, index = [0])




def fit_map(df, xy_range, dt_range, xy_nbins, dt_nbins, fit_function, nbins_S2e, S2e_range):

    df = df.loc[:,['X', 'Y', 'DT', 'S2e']]

    mask = (in_range(df.X, *xy_range) &
            in_range(df.Y, *xy_range) &
            in_range(df.DT, *dt_range)&
            in_range(df.S2e, *S2e_range))

    df = df[mask]

    x, y, dt, _ = df.values.T

    xy_bins = np.linspace(*xy_range, xy_nbins + 1)
    dt_bins = np.linspace(*dt_range, dt_nbins + 1)

    #add -1 so the indices go from 0 to 49, not from 1 to 50.
    map_df = df.assign( i = np.digitize(x, bins = xy_bins) - 1,
                        j = np.digitize(y, bins = xy_bins) - 1,
                        k = np.digitize(dt, bins = dt_bins) - 1)



    result = map_df.groupby(['k', 'i', 'j']).apply(fit_function, nbins_S2e)

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

    dt_binsize = (dt_range[1] - dt_range[0])/dt_nbins
    xy_binsize = (xy_range[1] - xy_range[0])/xy_nbins

    krmap =  krmap.assign( dt = dt_range[0] + (0.5+krmap.k)*dt_binsize,
                           x  = xy_range[0] + (0.5+krmap.i)*xy_binsize,
                           y  = xy_range[0] + (0.5+krmap.j)*xy_binsize )

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
        'dtbins'      : [krmap.k.unique().tolist()],
        'xbins'       : [krmap.i.unique().tolist()],
        'ybins'       : [krmap.j.unique().tolist()],
        'nbins_dt'    : dt_nbins,
        'nbins_x'     : xy_nbins,
        'nbins_y'     : xy_nbins,
        'map_shape'   : [df.shape],
        'map_extent'  : len(df)
    }


    return pd.DataFrame(metadata, index = [0]).T



def quick_gauss_fit(data, bins, sigma = False):
    """
    Histogram input data and fit it to a gaussian with the parameters
    automatically estimated.
    """
    y, x  = np.histogram(data, bins)
    x     = shift_to_bin_centers(x)
    seed  = gauss_seed(x, y)

    sigma = poisson_sigma(y) if sigma else None

    if sigma is None:
        f = fit(gauss, x, y, seed)
    else:
        f = fit(gauss, x, y, seed, sigma=sigma)

    if not np.all(f.values != seed):
        raise RuntimeError("all fit values should be different from seed")

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

        mask = in_range(df.time, t_start, t_end)
        df_slice = df[mask]
        dataframes.append(df_slice)

    return dataframes


def get_time_evol(df, col_name, run_number, bins_lt, x0, y0, shape, shape_size, p0, dtbins, low_DT, high_DT, bins_Ec, error = False, seed = None):

    """
    col_name needs to be the final corrected energy, for example, col_name = 'Ec_2' or whatever the name is

    dtbins should be a numpy.linspace to stablish dtrange and binning

    """

    mean = df.S2e.mean()
    std = df.S2e.std()/np.sqrt(len(df))

    kdst_in_region = select_lifetime_region(df, x0, y0, shape, shape_size)

    f = fit(expo, kdst_in_region.DT, kdst_in_region.S2e, p0)
    magnitudes, uncertainties = f.values, (f[2][0], f[2][1])

    lifetime = -magnitudes[1]
    ulifetime = uncertainties[1]

    e0 = magnitudes[0]
    e0u = uncertainties[0]

    dv, udv = compute_drift_v(kdst_in_region.DT.to_numpy(), dtbins, seed)


    f = quick_gauss_fit(df[col_name].values, bins_Ec, sigma = error)

    _, mu, sigma = f.values

    _, u_mu, u_sigma = f.errors

    R = np.sqrt(8*np.log(2))*(sigma/mu) #FWHM (krypton paper)
    u_R = R * (u_mu**2/mu**2 + u_sigma**2/sigma**2)**0.5

    mask = in_range(df.DT.values, low_DT, high_DT)
    s1e_cath = df[mask].S1e.values.mean()
    us1e_cath = df[mask].S1e.values.std()/np.sqrt(len(df))

    sqrtn = np.sqrt(len(df))

    t_evol = {'run_number' : run_number,
              'ts' : int(df.time.values[0]),
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
              's1w' : df.S1w.mean(),
              's1wu' : df.S1w.std()/sqrtn,
              's1e' : s1e_cath,
              's1eu' : us1e_cath,
              's1h': df.S1h.mean(),
              's1hu':df.S1h.std()/sqrtn,
              's2w':df.S2w.mean(),
              's2wu': df.S2w.std()/sqrtn,
              's2q': df.S2q.mean(),
              's2qu': df.S2q.std()/sqrtn,
              'Nsipm': df.Nsipm.mean(),
              'Nsipmu': df.Nsipm.std()/sqrtn,
              'Xrms': df.Xrms.mean(),
              'Xrmsu': df.Xrms.std()/sqrtn,
              'Yrms': df.Yrms.mean(),
              'Yrmsu': df.Yrms.std()/sqrtn,
              'Zrms': df.Zrms.mean(),
              'Zrmsu': df.Zrms.std()/sqrtn,
               }

    return pd.DataFrame(data = t_evol, index = [0])




def append_time_evol(dfs, col_name, run_number, bins_lt, x0, y0, shape, shape_size, p0, dtbins, low_DT, high_DT, bins_Ec, error = False, seed = None):

    df_tevols = []

    for df in dfs:

        t_evol = get_time_evol(df, col_name, run_number, bins_lt, x0, y0, shape, shape_size, p0, dtbins, low_DT, high_DT, bins_Ec, error = False, seed = None)

        df_tevols.append(t_evol)

    return pd.concat(df_tevols, ignore_index = True)



def save_map(name, efficiencies, krmap, metadata, t_evol):
    metadata.to_hdf(name, key = 'metadata', mode = 'w')

    with tb.open_file(name, "a") as file:
        df_writer(file, efficiencies, group_name = 'data', table_name = 'selection_efficiencies')
        df_writer(file, krmap, group_name = 'krmap', table_name = 'krmap')
        #df_writer(file, metadata, group_name = 'krmap_info', table_name = 'metadata')
        df_writer(file, t_evol, group_name = 't_evol', table_name = 't_evol')
