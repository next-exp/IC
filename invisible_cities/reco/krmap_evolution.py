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
from .. reco.icaro_components import get_fit_function_lt, select_fit_variables, transform_parameters # Won't work until previous PR are approved


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


def compute_drift_v(dtdata    : np.array,
                    nbins    : int,
                    dtrange   : Tuple[float, float],
                    seed     : Tuple[float, float, float, float],
                    detector : str)->Tuple[float, float]:

    '''
    Computes the drift velocity for a given distribution
    using the sigmoid function to get the cathode edge.

    Parameters
    ----------
    dtdata: array_like
        Values of DT coordinate.
    nbins: int (optional)
        The number of bins in the z coordinate for the binned fit.
    dtrange: length-2 tuple (optional)
        Fix the range in DT.
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

    y, x = np.histogram(dtdata, nbins, dtrange)
    x    = shift_to_bin_centers(x)

    if seed is None: seed = np.max(y), np.mean(dtrange), 0.5, np.min(y)

    # At the moment there is not NEXT-100 DB so this won't work for that geometry
    z_cathode = DB.DetectorGeo(detector).ZMAX[0]

    try:
        f   = fit(sigmoid, x, y, seed, sigma = poisson_sigma(y), fit_range = dtrange)

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
                            dtrange_dv  : List[float],
                            detector   : str)->DataFrame:

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
    dtrange_dv: List
        Range in Z-coordinate to use in the histogram for the drift
        velocity calculation.
    detector: string
        Used to get the cathode position from DB for the drift velocity
        computation.

    Returns
    -------
    pars: DataFrame
        Each column corresponds to the average value of a different parameter.
    '''

    # Computing E0, LT

    fit_func, seed = get_fit_function_lt(fittype)

    geo_correction_factor = e0_xy_correction(map        = emaps,
                                             norm_strat = NormStrategy.max)


    x, y = select_fit_variables(fittype, data) # Importar de previo PR

    corr = geo_correction_factor(data.X.to_numpy(), data.Y.to_numpy())

    y_corr = y -np.log(corr) if fittype == KrFitFunction.log_lin else y*corr

    # If we didn't care about using the specific fit function of the map computation,
    # this would be reduced to just the following:
    # y_corr = y*geo_correction_factor(data.X.to_numpy(), data.Y.to_numpy())

    fit_output, _, _, _ = fit(func        = fit_func, # Misma funcion que en el ajuste del mapa
                              x           = x,
                              y           = y_corr,
                              seed        = seed(x, y_corr),
                              full_output = False)

    if fittype == KrFitFunction.log_lin:

        par, err, cov = transform_parameters(fit_output)

        e0, lt = par
        e0u, ltu = err

    else:

        e0,  lt  = fit_output.values
        e0u, ltu = fit_output.errors

    # Computing Drift Velocity

    dv, dvu  = compute_drift_v(dtdata   = data.DT.to_numpy(),
                               nbins    = nbins_dv,
                               dtrange  = dtrange_dv,
                               seed     = None,
                               detector = detector)

    # Computing Resolution

    tot_corr_factor = apply_all_correction(maps       = emaps,
                                           apply_temp = False)

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

    # PARA METER CUT_EFFS_EVOLUTION: Esta función sólamente se aplica a la dst filtrada, y por lo tanto
    # no hay una colección de máscaras. Aquí sólo se mete el timestamp que hace referencia al tiempo del
    # rango temporal escogido para el tratamiento de la evolución temporal. Esto va dentro del bucle de
    # kr_time_evolution. Si queremos meter la evolución de eficiencias de cortes, yo la metería en esa
    # función y no en esta (aunque aquí en un primer momento pueda parecer más lógico) ya que en la otra
    # es posible filtrar la dst entera y aquí no según el flow de la función.


    return evol


def kr_time_evolution(ts         : np.array[float],
                      masks_time : List[np.array],
                      dst        : pd.DataFrame,
                      emaps      : pd.DataFrame,
                      fittype    : KrFitFunction,
                      nbins_dv   : int,
                      dtrange_dv  : Tuple[float, float],
                      detector   : str) -> pd.DataFrame:

    '''
    Computes some average parameters (e0, lt, drift v,
    S1w, S1h, S1e, S2w, S2h, S2e, S2q, Nsipm, Xrms, Yrms)
    for a given krypton distribution and for different time slices.
    Returns a DataFrame.

    Parameters
    ----------
    ts: np.array
        Sequence of central times for the different time slices.
    masks_time: list of boolean lists
        Allows dividing the distribution into time slices.
    dst: DataFrame
        Kdst distribution to analyze.
    emaps: correction map
        Allows geometrical correction of the energy.
    fittype: KrFitFunction
        Kind of fit to perform.
    nbins_dv: int
        Number of bins in Z-coordinate for doing the histogram to compute
        the drift velocity.
    dtrange_dv: Tuple
        Range in Z-coordinate for doing the histogram to compute the drift
        velocity.
    detector: string
        Used to get the cathode position from DB for the drift velocity
        computation.

    Returns
    -------
    evol_pars: pd.DataFrame
        DataFrame containing the parameters evolution. Each column corresponds
        to the average value for a given parameter. Each row corresponds to the
        parameters for a given time slice.
    '''

    # Create a DataFrame with the time slices (ts)
    mask_time_df = pd.DataFrame({'ts': ts})

    def compute_parameters(row, masks_time, dst, emaps, fittype, nbins_dv, dtrange_dv, detector):
        idx  = row.name
        mask = masks_time[idx]
        time = row['ts']

        # Select the data for the given mask
        sel_dst = dst[mask]

        # Compute the krypton parameters using the selected data
        pars = computing_kr_parameters(data      = sel_dst,
                                       ts        = time,
                                       emaps     = emaps,
                                       fittype   = fittype,
                                       nbins_dv  = nbins_dv,
                                       dtrange_dv = dtrange_dv,
                                       detector  = detector)



        if pars is None or pars.empty:
            return pd.Series(np.nan)  # Return nan Series if no parameters found

        if isinstance(pars, pd.DataFrame) and len(pars) == 1:
            pars = pars.iloc[0]  # Convert single-row DataFrame to Series

        return pd.Series(pars)


    # Apply the function to the previous df and compute krypton parameters for each time slice
    evol_pars = mask_time_df.apply(lambda row: compute_parameters(row, masks_time, dst, emaps,
                                                                  fittype, nbins_dv, dtrange_dv, detector),
                                   axis=1)

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


def all_krevol(map,      kdst,       r_fid,     nStimeprofile,
               mask_s1,  mask_s2,    mask_band, fittype,
               nbins_dv, dtrange_dv, detector):

    ''' COMPLETAR

    resumen: this function performs the whole proccess of evolution for all bins'''

    fid_sel   = (kdst.R < r_fid) & mask_s1 & mask_s2 & mask_band
    dstf      = kdst[fid_sel]
    t_start   = dstf.time.min()
    t_final   = dstf.time.max()
    ntimebins = np.max([int(np.floor((t_final - t_start) / nStimeprofile)), 1])

    ts, masks_time = get_time_series_df(ntimebins          = ntimebins,
                                        time_range         = (t_start, t_final),
                                        dst                = kdst)

    masks_timef    = [mask[fid_sel] for mask in masks_time]

    evol_table     = kr_time_evolution(ts                  = ts,
                                       masks_time          = masks_timef,
                                       dst                 = dstf,
                                       emaps               = map,
                                       fittype             = fittype,
                                       nbins_dv            = nbins_dv,
                                       dtrange_dv          = dtrange_dv,
                                       detector            = detector)

    evol_table_eff = cut_effs_evolution(masks_time         = masks_time,
                                        dst                = kdst,
                                        mask_s1            = mask_s1,
                                        mask_s2            = mask_s2,
                                        mask_band          = mask_band,
                                        evol_table         = evol_table)

    return evol_table_eff


def add_krevol(r_fid         : float,   # Esto sería para meter en la ciudad de ICARO, flow.map(funcion, args, out) etc
               nStimeprofile : int,
               **map_params):

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

    def add_krevol(map,      kdst,      mask_s1,
                   mask_s2,  mask_band, fittype,
                   nbins_dv, dtrange_dv, detector):

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
                                           dtrange_dv           = dtrange_dv,
                                           detector            = detector)

        evol_table_eff = cut_effs_evolution(masks_time         = masks_time,
                                            data               = kdst,
                                            mask_s1            = mask_s1,
                                            mask_s2            = mask_s2,
                                            mask_band          = mask_band,
                                            evol_table         = evol_table)

        return evol_table_eff

    return add_krevol
