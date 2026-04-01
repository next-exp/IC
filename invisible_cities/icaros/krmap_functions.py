import pandas as pd
import numpy  as np
from typing import Callable, Union

from .. core          .fit_functions import expo, gauss, fit
from .. io            .dst_io import load_dst, load_dsts, df_writer
from .. core          .core_functions import in_range, shift_to_bin_centers, weighted_mean_and_std
from .. core          .stat_functions import poisson_sigma
from .. types         .symbols import SelRegionMethod

from .. icaros        .lifetime_vdrift_functions import select_lifetime_region, compute_drift_v

import tables            as tb
from   scipy             import stats
from   scipy             import optimize
from   scipy.optimize import curve_fit

import itertools

"""

This script contains functions to compute empty maps, 3D maps from kdsts and merge them to create an 'uniform' map.
It also contains functions to get and store map info (metadata) and time evolution.

"""


def gauss_seed(x         : np.array,
               y         : np.array,
               sigma_rel : float = 0.02) -> tuple:
    """
    Estimate the seed for a gaussian fit to the input data.
    Parameters
    ----------
    x : np.array
      Gaussian distributed variable (in our case bin_centers of the energy histogram).
    y : np.array
      Frequency of each x value (counts of the energy histogram).
    sigma_rel : float
      Estimation of the relative sigma for x_max
    Returns
    -------
    seed : tuple
      Gaussian seed: area, mean value and standard deviation of the gaussian distributed data.
    """
    x_max = np.median(x)
    sigma = sigma_rel * x_max
    amp    = y.max()  * (2 * np.pi)**0.5 * sigma
    seed   = amp, x_max, sigma

    return seed



def create_empty_map(xy_range : tuple,
                     dt_range : tuple,
                     xy_nbins : int,
                     dt_nbins : int) -> pd.DataFrame:
    """
    - Uses every possible indices combination.
    - Creates a DataFrame filled with NaNs for each (i, j, k) combination.
    Parameters
    ----------
    xy_range : tuple
      Range in x and y (mm) inside which the map is being computed.
    dt_range : tuple
      Range in drift time (micro seconds) (same as z) inside which the map is being computed.
    xy_nbins : int
      Number of map bins for x and y.
    dt_nbins : int
      Number of map bins for drift time.
    Returns
    -------
    Nan_map : pd.DataFrame
      Dataframe filled with NaNs for each (i,j,k) combination.
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


def get_median(df        : pd.DataFrame,
               nbins_S2e : int = None) -> pd.DataFrame:
    """
    Computes the median value of a given variable
    -We consider that for len(df) < 3, it makes no sense to compute the std.
    -Kr energy resolution is ~4%, so the std for S2e median would be ~0.04/2.35.
    Parameters
    ----------
    df : pd.DataFrame
      Dataframe in which the median of the S2e values is being calculated.
    nbins_S2e : int
      Parameter that had to be added so it has the same structure as gaussian_fit
    Returns
    -------
    map : pd.DataFrame
      Dataframe containing 'nevents', 'mu', 'sigma', 'mu_error', 'sigma_error'
      of the input dataframe, where mu and mu_error are the median and the median error.
    """

    if len(df.S2e) < 5:
        sigma = (0.04/2.35)*df.S2e.median()
        mu = np.nan
    else:
        sigma = df.S2e.std()
        mu = df.S2e.median()

    return pd.DataFrame({
         'nevents': len(df.S2e),
         'mu': mu,
         'sigma': sigma,
         'mu_error': sigma/np.sqrt(len(df.S2e)),
         'sigma_error': np.nan}, index = [0])



def gaussian_fit(df         : pd.DataFrame,
                 nbins_S2e  : int,
                 min_events : int = 50) -> pd.DataFrame:
    """
    Given a dataframe, computes a gaussian fit of S2e if len(df) is equal or
    greater than min_events. If not, computes the median using get_median.
    Parameters
    ----------
    df : pd.DataFrame
      Dataframe in which the gaussian fit of the S2e values is being performed.
    nbins_S2e : int
      Number of bins to histogram S2e.
    Returns
    -------
    map : pd.DataFrame
      Dataframe containing 'nevents', 'mu', 'sigma', 'mu_error', 'sigma_error'
      of the input dataframe, where mu and mu_error are the mean and mean error
      of the fit.
    """

    if len(df)< min_events:
        return get_median(df)

    counts, bin_edges = np.histogram(df.S2e, nbins_S2e)
    bin_centers = shift_to_bin_centers(bin_edges)

    seed = gauss_seed(bin_centers, counts)

    try:
        f = fit(gauss, bin_centers, counts, seed = seed,  maxfev = 10000)


    except:

        return pd.DataFrame({'nevents': len(df),
                                 'mu': np.nan,
                                 'sigma': np.nan,
                                 'mu_error': np.nan,
                                 'sigma_error': np.nan}, index = [0])

    if not in_range(f.values[1], 0.5*df.S2e.median(), 1.5*df.S2e.median()):
        return get_median(df)


    return  pd.DataFrame({'nevents': len(df),
                            'mu': f.values[1],
                            'sigma': f.values[2],
                            'mu_error': f.errors[1],
                            'sigma_error': f.errors[2]}, index = [0])




def fit_map(df            : pd.DataFrame,
            xy_range      : tuple,
            dt_range      : tuple,
            xy_nbins      : int,
            dt_nbins      : int,
            S2e_range     : tuple,
            fit_function  : Callable,
            min_events    : int = 50,
            nbins_S2e     : int = None) -> pd.DataFrame:
    """
    For a given dataframe :
    - takes its x, y, dt and S2e values with their respective ranges.
    - calculates every possible indices combination to get i,j,k.
    - creates a map applying fit_function to data.
    Parameters
    ----------
    df : pd.DataFrame
      Input dataframe from which the map is being calculated.
    xy_range : tuple
      Range in x and y (mm) inside which the map is being computed.
    dt_range : tuple
      Range in drift time (micro seconds) (same as z) inside which the map is being computed.
    xy_nbins : int
      Number of map bins for x and y.
    dt_nbins : int
      Number of map bins for drift time.
    fit_function : function
      Function to fit S2e to.
    nbins_S2e: int
      Number of map bins for S2e.
    S2e_range: tuple
      Range in S2e (pe) inside which the map is being computed.
    Returns
    -------
    result : pd.DataFrame
      Map containing every possible indices combination (k, i, j) that have
      data and 'nevents', 'mu', 'sigma', 'mu_error', 'sigma_error' of the input
      dataframe after applying fit_function (note that it doesn't include spatial
      coordinates and if there is an index combination that doesn't contain data
      it won't appear in the output).
    """

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


def merge_maps(NaN_map : pd.DataFrame,
               map_3D  : pd.DataFrame) -> pd.DataFrame:

    """
    -When we do the 3D (data) maps we get a dataframe where some rows might be missing due to
    "lack of data". We combine this 3D data map with a "full map" filled with NaNs.
    We combine these two maps to ensure that all bins are present.
    -For bins with data, we have "repeated rows" with NaNs and data,
    we discard the NaN rows (doing .first()) to get a dataframe with full proper shape.
    Parameters
    ----------
    NaN_map : pd.DataFrame
      Map full of NaNs output of create_empty_map.
    map_3D  : pd.DataFrame
      3D data map output of fit_map.
    Returns
    -------
    full_map : pd.DataFrame
       Full map representing all bins (every possible indices combination), filled with
       data from fit_map and with NaNs in those bins where there is no data (note that
       spatial coordinates (dt, x, y) are still missing).
    """

    full_map = pd.concat([map_3D, NaN_map], ignore_index = True)
    full_map = full_map.groupby(['k', 'i','j']).first().reset_index()

    return full_map




def include_coordinates(krmap     : pd.DataFrame,
                        xy_range  : tuple,
                        dt_range  : tuple,
                        xy_nbins  : int,
                        dt_nbins  : int) -> pd.DataFrame:
    """
    This function adds (dt, x, y) spatial coordinates to an input map
    which would be the output of full_map as the center of x, y, dt bins
    computed given their respective ranges and number of bins.
    Parameters
    ----------
    krmap : pd.DataFrame
      Input map where the spatial coordinates are being added.
    xy_range : tuple
      Range in x and y where the krmap was computed and (x,y) coordinate limits.
    dt_range : tuple
      Range in drift time where the krmap was computed and dt coordinate limit.
    xy_nbins : int
      Number of bins in x and y to get the x,y coordinates.
    dt_nbins : int
      Number of bins in dt to get the dt coordinate.
    Returns
    -------
    krmap : pd.DataFrame
      Output map containing the same values as the input map with three more columns
      corresponding to dt, x and y spatial coordinates.
    """
    dt_binsize = (dt_range[1] - dt_range[0])/dt_nbins
    xy_binsize = (xy_range[1] - xy_range[0])/xy_nbins

    krmap =  krmap.assign( dt = dt_range[0] + (0.5+krmap.k)*dt_binsize,
                           x  = xy_range[0] + (0.5+krmap.i)*xy_binsize,
                           y  = xy_range[0] + (0.5+krmap.j)*xy_binsize
                           )

    return krmap


def regularize_map(krmap, mapshape):
    """
    -Function to regularize map in case there is any whole (NaN) inside the chamber volume.
    -It takes each bin in the map and, if there is a whole, searchs for non NaN and non zero
     neighbours and sets the mu value in the whole as the mean mu value of the neighbours.
    Parameters
    ----------
    krmap : pd.DataFrame
      Krypton map. Ideally the final "full" map after the whole process but it could be done
      at any step.
    mapshape : tuple
      Number of (k, i, j) bins in the map. Most common should be (10, 100, 100).
    Returns
    -------
    regularized_map : pd.DataFrame
      Full regularized map.

    """
    mu = krmap.mu.values.reshape(mapshape)
    for k in range(mu.shape[0]):
        for i in range(1, mu.shape[1] -1):
            for j in range(1, mu.shape[2] -1):
                if np.isnan(mu[k,i,j]):
                    vals = [mu[k,i+ii,j+jj] for ii in (-1, 0, 1) for jj in (-1, 0, 1) if not (ii==jj==0)]
                    if np.count_nonzero(np.isnan(vals))>3: continue
                    mu[k,i,j] = np.nanmean(vals)
    return krmap.assign(mu = mu.flatten())



def compute_3D_map(df           : pd.DataFrame,
                   xy_range     : tuple,
                   dt_range     : tuple,
                   xy_nbins     : int,
                   dt_nbins     : int,
                   S2e_range    : tuple,
                   fit_function : Callable,
                   min_events   : int = 50,
                   nbins_S2e    : int = None) -> pd.DataFrame:
    """
    Computes a full 3D map merging (with merge_maps) a map full of nans from create_empty_map
    with a 3D data map from fit_map and adding spatial coordinates using include_coordinates.
    Parameters
    ----------
    df  : pd.DataFrame
      Dataframe from which 3D (data) map is being calculated (input for fit_map).
    xy_range : tuple
      Range in x and y necessary to compute map full of nans, 3D map and to limit spatial
      coordinates.
    dt_range : tuple
      Range in drift time necessary to compute map full of nans, 3D maps and to limit
      spatial coordinates.
    xy_nbins : int
      Number of bins in x and y necessary to compute map full of nans, 3D maps and to set
      bin size for spatial coordinates.
    dt_nbins : int
      Number of bins in drift time necessary to compute map full of nans, 3D maps and to
      set bin size for dt spatial coordinate.
    fit_function : function
      Function to fit S2e to.
    nbins_S2e: int
      Number of map bins for S2e.
    S2e_range: tuple
      Range in S2e (pe) inside which the map is being computed.
    Returns
    -------
    full_map : pd.DataFrame
      Full map representing all bins (every possible indices combination), filled with
      data from fit_map,  with NaNs in those bins where there is no data and spatial
      dt, x, y coordinates.
    """
    ts = int(df.time.values.min())
    tf_slice = int(df.time.values.max())

    t_interval = tf_slice - ts

    t_hours = t_interval/3600 #from seconds to hours


    dt_binsize = (dt_range[1] - dt_range[0])/dt_nbins
    xy_binsize = (xy_range[1] - xy_range[0])/xy_nbins

    volume = (dt_binsize*xy_binsize*xy_binsize)*0.001 #from mm^3 to cm^3

    NaN_map = create_empty_map(xy_range, dt_range, xy_nbins, dt_nbins)
    map_3D_fit = fit_map(df, xy_range, dt_range, xy_nbins, dt_nbins, S2e_range, fit_function, min_events, nbins_S2e)

    map = merge_maps(NaN_map, map_3D_fit)
    coor_map = include_coordinates(map, xy_range, dt_range, xy_nbins, dt_nbins)

    full_map = coor_map.assign(density = coor_map['nevents']/volume/t_hours)
    #density in events*hour/cm^3

    full_map_regularized = regularize_map(full_map, (dt_nbins, xy_nbins, xy_nbins))

    return full_map_regularized




def compute_metadata(df       : pd.DataFrame,
                     krmap    : pd.DataFrame,
                     xy_range : tuple,
                     dt_range : tuple,
                     xy_nbins : int,
                     dt_nbins : int) -> pd.DataFrame:
    """
    Creates a dataframe including relevant map information (metadata).
    Parameters
    ----------
    df : pd.DataFrame
      Dataframe from which the krypton map was created.
    krmap : pd.Dataframe
      Final krypton map, output from compute_3D_map.
    xy_range : tuple
      Range in x and y used to create the map.
    dt_range : tuple
      Range in drift time used to create the map.
    xy_nbins : int
      Number of x, y bins used to create the map.
    dt_nbins : int
      Number of drift time bins used to create the map.
    Returns
    -------
    metadata : pd.DataFrame
      DataFrame containing information of relevant variables for the
      computation of the map.

    """

    metadata = {'rmax'        : df.R.max(),
                'zmax'        : df.Z.max(),
                'bin_size_dt' : (dt_range[1] - dt_range[0]) / dt_nbins,
                'bin_size_x'  : (xy_range[1] - xy_range[0]) / xy_nbins,
                'bin_size_y'  : (xy_range[1] - xy_range[0]) / xy_nbins,
                'dtbins'      : [krmap.k.unique().tolist()],
                'xbins'       : [krmap.i.unique().tolist()],
                'ybins'       : [krmap.j.unique().tolist()],
                'nbins_dt'    : dt_nbins,
                'nbins_x'     : xy_nbins,
                'nbins_y'     : xy_nbins,
                'xy_range'    : [xy_range],
                'dt_range'    : [dt_range],
                'map_shape'   : [df.shape],
                'map_extent'  : len(df)}


    return pd.DataFrame(metadata, index = [0]).T



def quick_gauss_fit(data  : np.array,
                    bins  : Union[int, np.array],
                    sigma : bool = False) -> Callable:
    """
    Histogram input data and fit it to a gaussian with the parameters
    automatically estimated.
    Parameters
    ----------
    data : np.array
      Input data to fit to a gaussian distribution.
    bins : int or np.array
      Either number of bins or a linspace of the bin edges.
    sigma: bool

    Returns
    -------
    f : Callable
      Gaussian fit output values from fit in invisible_cities/core/fit_functions.
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



def get_time_evol_single_slice(df           : pd.DataFrame,
                               col_name1    : str,
                               col_name2    : str,
                               run_number   : int,
                               x0           : float,
                               y0           : float,
                               shape        : SelRegionMethod,
                               shape_size   : float,
                               dtbins_dv    : np.array,
                               s1_DTrange   : tuple,
                               bins_Ec      : np.array,
                               error        : bool = False) -> pd.DataFrame:

    """
    Creates a dataframe including all the time evolution relevant parameters
    for a single time slice.

    Parameters
    ----------
    df : pd.DataFrame
      Input dataframe from which the time evolution parameters are being calculated.
    col_name1 : str
      Needs to be the name of the corrected energy obtained from applying
      the preliminary map.
      For example, col_name1 = 'Ec' (or whatever the name is).
    col_name2 : str
      Needs to be the name of the final corrected energy in the dataframe.
      For example, col_name2 = 'Ec_2' (or whatever the name is).
    run_number : int
      Number of the run that is being analyzed.
    x0 : float
      Point in x corresponding to the center of the circle or square.
    y0 : float
      Point in y corresponding to the center of the circle or square.
    shape : symbols/SelRegionMethod
      Geometric shape of the region, according to SelRegionMethod class.
    shape_size: float
      Either the value of the circle's radius or square half side.
    dtbins_dv : np.array
      Should be a numpy.linspace to stablish dtrange and binnnind.
    s1_DTrange : tuple
      Range in DT to get S1e values near cathode.
    bins_Ec : np.array
      Should be a numpy.linspace to stablish the corrected energy range.
    error : bool
      Error corresponding to sigma for quick_gauss_fit
    Returns
    -------
    t_evol : pd.DataFrame
      Dataframe including all the relevant time evolution parameters.
    """
    nevents = len(df.event.unique())
    sqrtn = np.sqrt(nevents)

    mean = df.S2e.mean()
    std = df.S2e.std()/sqrtn

    kdst_in_region = select_lifetime_region(df, x0, y0, shape, shape_size)

    #rough aproximation, should work fine
    p0 = (kdst_in_region.S2e.median(), -30000)

    try:
        f = fit(expo, kdst_in_region.DT, kdst_in_region.S2e, p0)
        magnitudes, uncertainties = f.values, (f[2][0], f[2][1])

        lifetime = -magnitudes[1]
        ulifetime = uncertainties[1]

        e0 = magnitudes[0]
        e0u = uncertainties[0]

    except:
        lifetime = np.nan
        ulifetime = np.nan

        e0 = np.nan
        e0u = np.nan

    dv, udv = compute_drift_v(kdst_in_region.DT.to_numpy(), dtbins_dv, seed = None)

    try:
        f = quick_gauss_fit(df[col_name1].values, bins_Ec, sigma = error)

        _, mu, sigma = f.values

        _, u_mu, u_sigma = f.errors

    except:

        median = get_median(df[col_name1])
        mu = median.mu
        sigma = median.sigma
        u_mu = median.mu_error
        u_sigma = median.sigma_error

    try:
        f2 = quick_gauss_fit(df[col_name2].values, bins_Ec, sigma = error)

        _, mu2, sigma2 = f2.values

        _, u_mu2, u_sigma2 = f2.errors

    except:

        median2 = get_median(df[col_name2])
        mu2 = median2.mu
        sigma2 = median2.sigma
        u_mu2 = median2.mu_error
        u_sigma2 = median2.sigma_error


    R = np.sqrt(8*np.log(2))*(sigma/mu) #FWHM (krypton paper)
    u_R = R * (u_mu**2/mu**2 + u_sigma**2/sigma**2)**0.5

    mask = in_range(df.DT.values, *s1_DTrange)
    s1e_cath = df[mask].S1e.values.mean()
    us1e_cath = df[mask].S1e.values.std()/np.sqrt(len(df))

    ts = int(df.time.values.min())
    tf_slice = int(df.time.values.max())

    t_interval = tf_slice - ts



    t_evol = {'run_number' : run_number,
              'ts' : ts,
              'nevents' : nevents,
              'neventsu': sqrtn,
              'rate' : nevents/t_interval,
              'rateu': sqrtn/t_interval,
              's2e': mean,
              's2eu' : std,
              'ec1' : mu,
              'ec1u': u_mu,
              'ec2' : mu2,
              'ec2u': u_mu2,
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
              's2h': df.S2h.mean(),
              's2hu': df.S2h.std()/sqrtn,
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




def get_time_evol(df          : pd.DataFrame,
                  slice_hours : float,
                  col_name1   : str,
                  col_name2   : str,
                  run_number  : int,
                  x0          : float,
                  y0          : float,
                  shape       : SelRegionMethod,
                  shape_size  : float,
                  dtbins_dv   : np.array,
                  s1_DTrange  : tuple,
                  bins_Ec     : np.array,
                  error       : bool = False) -> pd.DataFrame:
    """
    Splits input dataframe into time slices using 'time' column
    and then gets time evolution parameters.
    Creates a dataframe including all the time evolution relevant parameters.
    Parameters
    ----------
    df : pd.DataFrame
      Input dataframe that is going to be split in time slices and from which the
      time evolution parameters will be calculated.
    slice_hours : float
      Time interval (hours) in which the dataframe is being split
    The rest of parameters are the same that those in get_time_evol_single_slice
    Returns
    -------
    t_evols : pd.DataFrame
      Dataframe containing time evolution parameters for each time slice.
    """

    slice_seconds = slice_hours * 3600

    t_min = df.time.min()
    t_max = df.time.max()

    time_bins = np.arange(t_min, t_max + slice_seconds, slice_seconds)

    t_slice = np.digitize(df.time, time_bins)

    t_evols  = df.groupby(t_slice).apply(get_time_evol_single_slice,
                                         col_name1,
                                         col_name2,
                                         run_number,
                                         x0,
                                         y0,
                                         shape,
                                         shape_size,
                                         dtbins_dv,
                                         s1_DTrange,
                                         bins_Ec,
                                         error)

    return t_evols.reset_index(drop = True)




def save_map(name          : str,
             efficiencies  : pd.DataFrame,
             krmap         : pd.DataFrame,
             metadata      : pd.DataFrame,
             t_evol        : pd.DataFrame):
    """
    Saves efficiencies, krypton map, metadata and time evolution in the same hdf file.
    ...still have work to do
    Parameters
    ----------
    name : str
      Name of the hdf file.
    efficiencies : pd.DataFrame
      Efficiencies dataframe.
    krmap : pd.DataFrame
      3D krypton map, output of compute_3D_map.
    metadata : pd.DataFrame
      Metadata dataframe.
    t_evol : pd.DataFrame
      Time evolution dataframe.
    Returns
    -------
    Hdf file containing in each node each one of the inputs.
    """

    metadata.to_hdf(name, key = 'metadata', mode = 'a')

    with tb.open_file(name, "a") as file:
        df_writer(file, efficiencies, group_name = 'data', table_name = 'selection_efficiencies')
        df_writer(file, krmap, group_name = 'krmap', table_name = 'krmap')
        #df_writer(file, metadata, group_name = 'krmap_info', table_name = 'metadata')
        df_writer(file, t_evol, group_name = 't_evol', table_name = 't_evol')
