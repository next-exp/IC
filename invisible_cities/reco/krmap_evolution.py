import numpy  as np
import pandas as pd

from   typing                 import List, Tuple, Optional, Callable
from   pandas                 import DataFrame

from .. types.symbols         import NormStrategy
from .. types.symbols         import KrFitFunction # Won't work until previous PR are approved
from .. core.fit_functions    import fit, gauss
from .. core.core_functions   import in_range, shift_to_bin_centers
from .. reco.corrections      import get_normalization_factor
from .. reco.corrections      import correct_geometry_
from .. reco.corrections      import maps_coefficient_getter
from .. reco.corrections      import apply_all_correction
from .. core.stat_functions   import poisson_sigma
from .. database              import load_db  as  DB
from .. reco.icaro_components import get_fit_function_lt # Won't work until previous PR are approved


def sigmoid(x          : np.array,
            scale      : float,
            inflection : float,
            slope      : float,
            offset     : float)->np.array:
    '''
    Sigmoid function, it computes the sigmoid of the input array x using the specified
    parameters for scaling, inflection point, slope, and offset.

    Parameters
    ----------
    x : np.array
        The input array.
    scale : float
        The scaling factor determining the maximum value of the sigmoid function.
    inflection : float
        The x-value of the sigmoid's inflection point (where the function value is half of the scale).
    slope : float
        The slope parameter that controls the steepness of the sigmoid curve.
    offset : float
        The vertical offset added to the sigmoid function.

    Returns
    -------
    np.array
        Array of computed sigmoid values for x array.
    '''

    sigmoid = scale / (1 + np.exp(-slope * (x - inflection))) + offset

    return sigmoid


def gauss_seed(x         : np.array,
               y         : np.array,
               sigma_rel : Optional[int] = 0.05):

    '''
    This function estimates the seed for a gaussian fit.

    Parameters
    ----------
    x: np.array
        Data to fit.
    y: int
        Number of bins for the histogram.
    sigma_rel (Optional): int
        Relative error, default 5%.

    Returns
    -------
    seed: Tuple
        Tuple with the seed estimation.
    '''

    # Looks for the higher Y value and takes its corresponding X value
    # as the center of the Gaussian. Based on the "sigma_rel" parameter,
    # applies some error to that X value and provides the estimation for the
    # amplitude, the center and the sigma.

    y_max  = np.argmax(y)
    x_max  = x[y_max]
    sigma  = sigma_rel * x_max
    amp    = y_max * (2 * np.pi)**0.5 * sigma * np.diff(x)[0]
    seed   = amp, x_max, sigma

    return seed


def resolution(values : np.array,
               errors : np.array):

    '''
    Computes the resolution (% FWHM) from the Gaussian parameters.

    Parameters
    ----------
    values: np.array
        Gaussian parameters: amplitude, center, and sigma.
    errors: np.array
        Uncertainties for the Gaussian parmeters.

    Returns
    -------
    res: float
        Resolution.
    ures: float
        Uncertainty of resolution.
    '''

    amp  ,   mu,   sigma = values
    u_amp, u_mu, u_sigma = errors

    res  = 235.48 * sigma/mu
    ures = res * (u_mu**2/mu**2 + u_sigma**2/sigma**2)**0.5

    return res, ures


def quick_gauss_fit(data : np.array,
                    bins : int):

    '''
    This function histograms input data and then fits it to a Gaussian.

    Parameters
    ----------
    data: np.array
        Data to fit.
    bins: int
        Number of bins for the histogram.

    Returns
    -------
    fit_output: FitFunction
        Object containing the fit results
    '''

    y, x  = np.histogram(data, bins)
    x     = shift_to_bin_centers(x)
    seed  = gauss_seed(x, y)

    fit_result = fit(gauss, x, y, seed)

    return fit_result


def get_number_of_time_bins(nStimeprofile : int,
                            tstart        : int,
                            tfinal        : int)->int:

    '''
    Computes the number of time bins to use for a given time step
    in seconds.

    Parameters
    ----------
    nStimeprofile: int
        Time step in seconds.
    tstart: int
        Initial timestamp for the dataset.
    tfinal: int
        Final timestamp for the dataset.

    Returns
    -------
    ntimebins: int
        Number of time bins.
    '''

    ntimebins = int(np.floor((tfinal - tstart) / nStimeprofile))
    ntimebins = np.max([ntimebins, 1])

    return ntimebins


def get_time_series_df(ntimebins  : int,
                       time_range : Tuple[float, float],
                       dst        : DataFrame)->Tuple[np.array, List[np.array]]:

    '''
    Given a dst this function returns a time series (ts) and a list of masks which are used to divide
    the dst in time intervals.

    Parameters
    ----------
        ntimebins : int
            Number of time bins
        time_range : Tuple
            Time range
        dst : pd.DataFrame
            DataFrame

    Returns
    -------
        A Tuple with:
            np.array       : The time series
            List[np.array] : The list of masks to get the events for each time series.
    '''

    modified_right_limit = np.nextafter(time_range[-1], np.inf)
    time_bins            = np.linspace(time_range[0], modified_right_limit, ntimebins+1)
    masks                = np.array([in_range(dst['time'].to_numpy(), time_bins[i], time_bins[i + 1]) for i in range(ntimebins)])

    return shift_to_bin_centers(time_bins), masks


def compute_drift_v(zdata    : np.array,
                    nbins    : int,
                    zrange   : Tuple[float, float],
                    seed     : Tuple[float, float, float, float],
                    detector : str)->Tuple[float, float]:

    '''
    Computes the drift velocity for a given distribution
    using the sigmoid function to get the cathode edge.

    Parameters
    ----------
    zdata: array_like
        Values of Z coordinate.
    nbins: int (optional)
        The number of bins in the z coordinate for the binned fit.
    zrange: length-2 tuple (optional)
        Fix the range in z.
    seed: length-4 tuple (optional)
        Seed for the fit.
    detector: string (optional)
        Used to get the cathode position from DB.

    Returns
    -------
    dv: float
        Drift velocity.
    dvu: float
        Drift velocity uncertainty.
    '''

    y, x = np.histogram(zdata, nbins, zrange)
    x    = shift_to_bin_centers(x)

    if seed is None: seed = np.max(y), np.mean(zrange), 0.5, np.min(y)

    # At the moment there is not NEXT-100 DB so this won't work for that geometry
    z_cathode = DB.DetectorGeo(detector).ZMAX[0]

    try:
        f   = fit(sigmoid, x, y, seed, sigma = poisson_sigma(y), fit_range = zrange)

        par = f.values
        err = f.errors

        dv  = z_cathode/par[1]
        dvu = dv/par[1] * err[1]

    except RuntimeError:
        print("WARNING: Sigmoid fit for dv computation fails. NaN value will be set in its place.")
        dv, dvu = np.nan, np.nan

    return dv, dvu


def e0_xy_correction(map        : pd.DataFrame,
                     norm_strat : NormStrategy)->Callable:

    '''
    Provides the function to compute only the geometrical corrections.

    Parameters
    ----------
    map: pd.DataFrame
        Map containing the corrections.
    norm_strat:
        Normalization strategy used when correcting the energy.

    Returns
    -------
        A function to compute geometrical corrections given a hit's X,Y position.
    '''

    normalization   = get_normalization_factor(map        , norm_strat)
    get_xy_corr_fun = maps_coefficient_getter (map.mapinfo, map.e0)

    def geo_correction_factor(x : np.array,
                              y : np.array):
        return correct_geometry_(get_xy_corr_fun(x,y))*normalization

    return geo_correction_factor


def computing_kr_parameters(data       : DataFrame,
                            ts         : float,
                            emaps      : pd.DataFrame,
                            fittype    : KrFitFunction,
                            nbins_dv   : int,
                            zrange_dv  : List[float, float],
                            detector   : str,
                            norm_strat : NormStrategy,
                            norm_value : float)->DataFrame: # REVISAR NORM_STRAT Y NORM_VALUE

    '''
    Computes some average parameters (e0, lt, drift v, energy
    resolution, S1w, S1h, S1e, S2w, S2h, S2e, S2q, Nsipm, 'Xrms, Yrms)
    for a given krypton distribution.

    Parameters
    ----------
    data: DataFrame
        Kdst distribution to analyze.
    ts: float
        Central time of the distribution.
    emaps: correction map
        Allows geometrical correction of the energy.
    fittype: KrFitFunction
        Kind of fit to perform
    nbins_dv: int
        Number of bins in Z-coordinate to use in the histogram for the
        drift velocity calculation.
    zrange_dv: List
        Range in Z-coordinate to use in the histogram for the drift
        velocity calculation.
    detector: string
        Used to get the cathode position from DB for the drift velocity
        computation.
    norm_strat: NormStrategy
        Normalization strategy to follow.
    norm_value: float
        Energy value to normalize to.

    Returns
    -------
    pars: DataFrame
        Each column corresponds to the average value of a different parameter.
    '''

    # Computing E0, LT

    geo_correction_factor = e0_xy_correction(map        = emaps,
                                             norm_strat = norm_strat) # PREGUNTAR POR ESTRATEGIA

    fit_func, seed = get_fit_function_lt(fittype)

    x           = data.DT,
    y           = data.S2e.to_numpy()*geo_correction_factor(data.X.to_numpy(), data.Y.to_numpy())

    fit_output, _, _, _ = fit(func        = fit_func, # Misma funcion que en el ajuste del mapa
                              x           = x,
                              y           = y,
                              seed        = seed(x, y),
                              full_output = False)

    e0,  lt  = fit_output.values
    e0u, ltu = fit_output.errors

    # Computing Drift Velocity

    dv, dvu  = compute_drift_v(zdata    = data.Z.to_numpy(),
                               nbins    = nbins_dv,
                               zrange   = zrange_dv,
                               detector = detector)

    # Computing Resolution

    tot_corr_factor = apply_all_correction(maps       = emaps,
                                           apply_temp = False,
                                           norm_strat = norm_strat,
                                           norm_value = norm_value)

    nbins = int((len(data.S2e))**0.5) # Binning as a function of nevts. Should we change it?

    f = quick_gauss_fit(data.S2e.to_numpy()*tot_corr_factor(data.X.   to_numpy(),
                                                            data.Y.   to_numpy(),
                                                            data.Z.   to_numpy(),
                                                            data.time.to_numpy()),
                        bins = nbins)

    par, err     = f.values, f.errors
    res, err_res = resolution(values = par,
                              errors = err)

    # Averages of parameters

    parameters = ['S1w', 'S1h', 'S1e',
                  'S2w', 'S2h', 'S2e', 'S2q',
                  'Nsipm', 'Xrms', 'Yrms']

    mean_d, var_d = {}, {}

    for parameter in parameters:

        data_value        = getattr(data, parameter)
        mean_d[parameter] = np.mean(data_value)
        var_d [parameter] = (np.var(data_value)/len(data_value))**0.5

    # Creating parameter evolution table

    evol = DataFrame({'ts'   : [ts]             ,
                      'e0'   : [e0]             , 'e0u'   : [e0u]           ,
                      'lt'   : [lt]             , 'ltu'   : [ltu]           ,
                      'dv'   : [dv]             , 'dvu'   : [dvu]           ,
                      'resol': [res]            , 'resolu': [err_res]       ,
                      's1w'  : [mean_d['S1w']]  , 's1wu'  : [var_d['S1w']]  ,
                      's1h'  : [mean_d['S1h']]  , 's1hu'  : [var_d['S1h']]  ,
                      's1e'  : [mean_d['S1e']]  , 's1eu'  : [var_d['S1e']]  ,
                      's2w'  : [mean_d['S2w']]  , 's2wu'  : [var_d['S2w']]  ,
                      's2h'  : [mean_d['S2h']]  , 's2hu'  : [var_d['S2h']]  ,
                      's2e'  : [mean_d['S2e']]  , 's2eu'  : [var_d['S2e']]  ,
                      's2q'  : [mean_d['S2q']]  , 's2qu'  : [var_d['S2q']]  ,
                      'Nsipm': [mean_d['Nsipm']], 'Nsipmu': [var_d['Nsipm']],
                      'Xrms' : [mean_d['Xrms']] , 'Xrmsu' : [var_d['Xrms']] ,
                      'Yrms' : [mean_d['Yrms']] , 'Yrmsu' : [var_d['Yrms']]})

    return evol


def kr_time_evolution(ts         : np.array[float],
                      masks_time : List[np.array],
                      dst        : pd.DataFrame,
                      emaps      : pd.DataFrame,
                      fittype    : KrFitFunction,
                      nbins_dv   : int,
                      zrange_dv  : Tuple[float, float],
                      detector   : str,
                      norm_strat : NormStrategy,
                      norm_value : float)->DataFrame:


    '''
    Computes some average parameters (e0, lt, drift v,
    S1w, S1h, S1e, S2w, S2h, S2e, S2q, Nsipm, 'Xrms, Yrms)
    for a given krypton distribution and for different time slices.
    Returns a DataFrame.

    Parameters
    ----------
    ts: np.array
        Sequence of central times for the different time slices.
    masks_time: list of boolean lists
        Allows dividing the distribution into time slices.
    data: DataFrame
        Kdst distribution to analyze.
    emaps: correction map
        Allows geometrical correction of the energy.
    fittype: KrFitFunction
        Kind of fit to perform.
    nbins_dv: int
        Number of bins in Z-coordinate for doing the histogram to compute
        the drift velocity.
    zrange_dv: int
        Range in Z-coordinate for doing the histogram to compute the drift
        velocity.
    detector: string
        Used to get the cathode position from DB for the drift velocity
        computation.
    norm_strat: NormStrategy
        Normalization strategy to follow.
    norm_value: float
        Energy value to normalize to.

    Returns
    -------
    evol_pars: pd.DataFrame
        Dataframe containing the parameters evolution. Each column corresponds
        to the average value for a given parameter. Each row corresponds to the
        parameters for a given time slice.
    '''

    frames = []

    for index in range(len(masks_time)):

        sel_dst = dst[masks_time[index]]
        pars    = computing_kr_parameters(data       = sel_dst,
                                          ts         = ts[index],
                                          emaps      = emaps,
                                          fittype    = fittype,
                                          nbins_dv   = nbins_dv,
                                          zrange_dv  = zrange_dv,
                                          detector   = detector,
                                          norm_strat = norm_strat,
                                          norm_value = norm_value)
        frames.append(pars)

    evol_pars = pd.concat(frames, ignore_index=True)

    return evol_pars


def cut_effs_evolution(masks_time : List[np.array],
                       dst        : pd.DataFrame,
                       mask_s1    : np.array,
                       mask_s2    : np.array,
                       mask_band  : np.array,
                       evol_table : pd.DataFrame):

    '''
    Computes the efficiencies in time evolution for different time slices.
    Returns the input DataFrame updated with S1eff, S2eff, Bandeff.

    Parameters
    ----------
    masks_time: list of time masks
        Masks which divide the data into time slices.
    data: pd.DataFrame
        kdst data.
    mask_s1: np.array
        Mask of S1 cut.
    mask_s2: np.array
        Mask of S2 cut.
    mask_band: np.array
        Mask of band cut.
    evol_table: pd.DataFrame
        Table of Kr evolution parameters.

    Returns
    -------
    evol_table_updated: pd.DataFrame
        Kr evolution parameters table updated with efficiencies.
    '''

    len_ts = len(masks_time)

    n0     = np.zeros(len_ts)
    nS1    = np.zeros(len_ts)
    nS2    = np.zeros(len_ts)
    nBand  = np.zeros(len_ts)

    for index in range(len_ts):

        time_mask    = masks_time[index]
        nS1mask      = time_mask & mask_s1
        nS2mask      = nS1mask   & mask_s2
        nBandmask    = nS2mask   & mask_band

        n0   [index] = dst[time_mask].event.nunique()
        nS1  [index] = dst[nS1mask]  .event.nunique()
        nS2  [index] = dst[nS2mask]  .event.nunique()
        nBand[index] = dst[nBandmask].event.nunique()

    evol_table_updated = evol_table.assign(S1eff   = nS1   / n0,
                                           S2eff   = nS2   / nS1,
                                           Bandeff = nBand / nS2)

    return evol_table_updated


def add_krevol(r_fid         : float,   # Esto sería para meter en la ciudad de ICARO, flow.map(funcion, args, out) etc
               nStimeprofile : int,
               **map_params):      # PREGUNTA: Pongo aquí explícitamente todos los argumentos? O se pueden meter con un diccionario?

    '''
    Adds the time evolution dataframe to the map.

    Parameters
    ---------
    r_fid: float
        Maximum radius for fiducial sample.
    nStimeprofile: int
        Number of seconds for each time bin.
    map_params: dict
        Dictionary containing the config file variables.

    Returns
    ---------
    Function which takes as input map, kr_data, and kr_mask
    and returns the time evolution.
    '''

    def add_krevol(map,      kdst,      mask_s1, # Más de lo mismo con la pregunta anterior
                   mask_s2,  mask_band, fittype,
                   nbins_dv, zrange_dv, detector):

        fid_sel   = (kdst.R < r_fid)  & mask_s1 & mask_s2 & mask_band
        dstf      = kdst[fid_sel]
        min_time  = dstf.time.min()
        max_time  = dstf.time.max()

        ntimebins      = get_number_of_time_bins(nStimeprofile = nStimeprofile,
                                                 tstart        = min_time,
                                                 tfinal        = max_time)

        ts, masks_time = get_time_series_df(ntimebins          = ntimebins,
                                            time_range         = (min_time, max_time),
                                            dst                = kdst)

        masks_timef    = [mask[fid_sel] for mask in masks_time]

        evol_table     = kr_time_evolution(ts                  = ts,
                                           masks_time          = masks_timef,
                                           dst                 = dstf,
                                           emaps               = map,
                                           fittype             = fittype,
                                           nbins_dv            = nbins_dv,
                                           zrange_dv           = zrange_dv,
                                           detector            = detector)

        evol_table_eff = cut_effs_evolution(masks_time         = masks_time,
                                            data               = kdst,
                                            mask_s1            = mask_s1,
                                            mask_s2            = mask_s2,
                                            mask_band          = mask_band,
                                            evol_table         = evol_table)

        return evol_table_eff

    return add_krevol
