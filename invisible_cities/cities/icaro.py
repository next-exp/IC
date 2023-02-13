"""
-----------------------------------------------------------------------
                                Icaro
-----------------------------------------------------------------------


"""

import numpy as np
import pandas as pd
import tables as tb

from .. reco                import tbl_functions as tbl
from .. reco.corrections    import ASectorMap
from .. reco.corrections    import maps_coefficient_getter
from .. reco.corrections    import correct_geometry_
from .. reco.corrections    import norm_strategy
from .. reco.corrections    import get_normalization_factor
from .. reco.corrections    import apply_all_correction
from .. io.krmaps_io        import write_krmap
from .. io.krmaps_io        import write_krevol
from .. io.krmaps_io        import write_mapinfo
from .. core                import fit_functions as fitf
from .. core.core_functions import in_range
from .. core.core_functions import shift_to_bin_centers

from .. dataflow          import dataflow as fl

from .. types.ic_types    import Measurement
from .. types.ic_types    import MasksContainer

from .  components import city
from .  components import print_numberofevents
from .  components import dst_from_files
from .  components import quality_check
from .  components import kr_selection
from .  components import map_builder
from .  components import add_krevol

from typing import Tuple
from typing import List
from pandas import DataFrame


@city
def icaro(files_in, file_out, compression, event_range,
          detector_db, run_number, bootstrap, quality_ranges,
          band_sel_params, map_params,
          r_fid, nStimeprofile, x_range, y_range):


    quality_check_before = fl.map(quality_check(quality_ranges),
                                  args = "input_data",
                                  out  = "checks")
    quality_check_after  = fl.map(quality_check(quality_ranges),
                                  args = "kr_data",
                                  out  = "checks")

    kr_selection_map     = fl.map(kr_selection(bootstrap, band_sel_params),
                                  args = "input_data",
                                  out  = ("kr_data", "kr_mask"))

    map_builder_map      = fl.map(map_builder(detector_db, run_number, map_params),
                                  args = "kr_data",
                                  out  = ("map_info", "map")                      )

    add_krevol_map       = fl.map(add_krevol(r_fid, nStimeprofile),
                                  args = ("map", "kr_data", "kr_mask"),
                                  out  = "evolution"           )

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        # Define writers...
        write_krmap_sink      = fl.sink(  write_krmap(h5out), args=("run_number", "event_number", "timestamp"))
        write_krevol_sink     = fl.sink( write_krevol(h5out), args="pointlike_event")
        write_mapinfo_sink    = fl.sink(write_mapinfo(h5out), args=("event_number", "pmap_passed"))

        return fl.push(source = dst_from_files(files_in, "DST", "Events")      ,
                       pipe   = fl.pipe( print_numberofevents                  ,
                                         quality_check_before                  ,
                                         kr_selection_map                      ,
                                         quality_check_after                   ,
                                         print_numberofevents                  ,
                                         map_builder_map                       ,
                                         add_krevol_map                        ,
                                         fl.fork(write_krmap_sink              ,
                                                 write_krevol_sink             ,
                                                 write_mapinfo_sink             )),
                       result = None)


def add_krevol(r_fid        : float,
               nStimeprofile: int):
    """
    Adds time evolution dataframe to the map
    Parameters
    ---------
    r_fid: float
        Maximum radius for fiducial sample
    nStimeprofile: int
        Number of seconds for each time bin
    Returns
    ---------
    Function which takes as input map, kr_data, and kr_mask
    and returns the time evolution
    """

    def add_krevol(map, kr_data, kr_mask):
        fmask     = (kr_data.R < r_fid) & kr_mask.s1 & kr_mask.s2 & kr_mask.band
        dstf      = kr_data[fmask]
        min_time  = dstf.time.min()
        max_time  = dstf.time.max()
        ntimebins = get_number_of_time_bins(nStimeprofile = nStimeprofile,
                                            tstart        = min_time,
                                            tfinal        = max_time)

        ts, masks_time = get_time_series_df(time_bins  = ntimebins,
                                            time_range = (min_time, max_time),
                                            dst        = kr_data)

        masks_timef    = [mask[fmask] for mask in masks_time]
        pars           = kr_time_evolution(ts         = ts,
                                           masks_time = masks_timef,
                                           dst        = dstf,
                                           emaps      = map)

        pars_ec        = cut_time_evolution(masks_time = masks_time,
                                            dst        = kr_data,
                                            masks_cuts = kr_mask,
                                            pars_table = pars)

        e0par       = np.array([pars['e0'].mean(), pars['e0'].var()**0.5])
        ltpar       = np.array([pars['lt'].mean(), pars['lt'].var()**0.5])
        print("    Mean core E0: {0:.1f}+-{1:.1f} pes".format(*e0par))
        print("    Mean core Lt: {0:.1f}+-{1:.1f} mus".format(*ltpar))

        return pars_ec

    return add_krevol

def get_number_of_time_bins(nStimeprofile: int,
                            tstart       : int,
                            tfinal       : int )->int:
    """
    Computes the number of time bins to compute for a given time step
    in seconds.

    Parameters
    ----------
    nStimeprofile: int
        Time step in seconds
    tstart: int
        Initial timestamp for the data set
    tfinal: int
        Final timestamp for the data set

    Returns
    -------
    ntimebins: int
        Number of bins
    """
    ntimebins = int( np.floor( ( tfinal - tstart) / nStimeprofile) )
    ntimebins = np.max([ntimebins, 1])

    return ntimebins


def get_time_series_df(time_bins    : int,
                       time_range   : Tuple[float, float],
                       dst          : DataFrame,
                       time_column  : str = 'time')->Tuple[np.array, List[np.array]]:
    """
    Given a dst (DataFrame) with a time column specified by the name time,
    this function returns a time series (ts) and a list of masks which are used to divide
    the event in time tranches.
    More generically, one can produce a "time series" using any column of the dst
    simply specifying time_column = ColumName
        Parameters
        ----------
            time_bins
                Number of time bines.
            time_range
                Time range.
            dst
                A Data Frame
            time_column
            A string specifyng the dst column to be divided in time slices.
        Returns
        -------
            A Tuple with:
            np.array       : This is the ts vector
            List[np.array] : This are the list of masks defining the events in the time series.
    """
    #Add small number to right edge to be included with in_range function
    modified_right_limit = np.nextafter(time_range[-1], np.inf)
    ip = np.linspace(time_range[0], modified_right_limit, time_bins+1)
    masks = np.array([in_range(dst[time_column].values, ip[i], ip[i + 1]) for i in range(len(ip) -1)])
    return shift_to_bin_centers(ip), masks


def kr_time_evolution(ts         : np.array,
                      masks_time : List[np.array],
                      dst        : DataFrame,
                      emaps      : ASectorMap,
                      zslices_lt : int                 = 50,
                      zrange_lt  : Tuple[float,float]  = (0, 550),
                      nbins_dv   : int                 = 35,
                      zrange_dv  : Tuple[float, float] = (500, 625),
                      detector   : str                 = 'new')->DataFrame:
    """
    Computes some average parameters (e0, lt, drift v,
    S1w, S1h, S1e, S2w, S2h, S2e, S2q, Nsipm, 'Xrms, Yrms)
    for a given krypton distribution and for different time slices.
    Returns a DataFrame.
    Parameters
    ----------
    ts: np.array of floats
        Sequence of central times for the different time slices.
    masks_time: list of boolean lists
        Allows dividing the distribution into time slices.
    data: DataFrame
        Kdst distribution to analyze.
    emaps: correction map
        Allows geometrical correction of the energy.
    z_slices: int (optional)
        Number of Z-coordinate bins for doing the exponential fit to compute
        the lifetime.
    zrange_lt: length-2 tuple (optional)
        Number of Z-coordinate range for doing the exponential fit to compute
        the lifetime.
    nbins_dv: int (optional)
        Number of bins in Z-coordinate for doing the histogram to compute
        the drift velocity.
    zrange_dv: int (optional)
        Range in Z-coordinate for doing the histogram to compute the drift
        velocity.
    detector: string (optional)
        Used to get the cathode position from DB for the drift velocity
        computation.
    Returns
    -------
    pars: DataFrame
        Each column corresponds to the average value for a given parameter.
        Each row corresponds to the parameters for a given time slice.
    """

    frames = []
    for index in range(len(masks_time)):
        sel_dst = dst[masks_time[index]]
        pars    = computing_kr_parameters(sel_dst, ts[index],
                                          emaps,
                                          zslices_lt, zrange_lt,
                                          nbins_dv, zrange_dv,
                                          detector)
        frames.append(pars)

    total_pars = pd.concat(frames, ignore_index=True)

    return total_pars


def e0_xy_correction(map        : ASectorMap                         ,
                     norm_strat : norm_strategy   = norm_strategy.max):
    """
    Temporal function to perfrom IC geometric corrections only
    """
    normalization   = get_normalization_factor(map        , norm_strat)
    get_xy_corr_fun = maps_coefficient_getter (map.mapinfo, map.e0)
    def geo_correction_factor(x : np.array,
                              y : np.array) -> np.array:
        return correct_geometry_(get_xy_corr_fun(x,y))* normalization
    return geo_correction_factor


def computing_kr_parameters(data       : DataFrame,
                            ts         : float,
                            emaps      : ASectorMap,
                            zslices_lt : int,
                            zrange_lt  : Tuple[float,float]  = (0, 550),
                            nbins_dv   : int                 = 35,
                            zrange_dv  : Tuple[float, float] = (500, 625),
                            detector   : str                 = 'new')->DataFrame:

    """
    Computes some average parameters (e0, lt, drift v, energy
    resolution, S1w, S1h, S1e, S2w, S2h, S2e, S2q, Nsipm, 'Xrms, Yrms)
    for a given krypton distribution. Returns a DataFrame.
    Parameters
    ----------
    data: DataFrame
        Kdst distribution to analyze.
    ts: float
        Central time of the distribution.
    emaps: correction map
        Allows geometrical correction of the energy.
    xr_map, yr_map: length-2 tuple
        Set the X/Y-coordinate range of the correction map.
    nx_map, ny_map: int
        Set the number of X/Y-coordinate bins for the correction map.
    zslices_lt: int
        Number of Z-coordinate bins for doing the exponential fit to compute
        the lifetime.
    zrange_lt: length-2 tuple (optional)
        Number of Z-coordinate range for doing the exponential fit to compute
        the lifetime.
    nbins_dv: int (optional)
        Number of bins in Z-coordinate for doing the histogram to compute
        the drift velocity.
    zrange_dv: int (optional)
        Range in Z-coordinate for doing the histogram to compute the drift
        velocity.
    detector: string (optional)
        Used to get the cathode position from DB for the drift velocity
        computation.
    Returns
    -------
    pars: DataFrame
        Each column corresponds to the average value of a different parameter.
    """

    ## lt and e0
    geo_correction_factor = e0_xy_correction(map =  emaps                         ,
                                             norm_strat = norm_strategy.max)

    _, _, fr = fitf.fit_lifetime_profile(data.Z,
                                         data.S2e.values*geo_correction_factor(
                                            data.X.values,
                                            data.Y.values),
                                         zslices_lt, zrange_lt)
    e0,  lt  = fr.par
    e0u, ltu = fr.err

    ## compute drift_v
    dv, dvu  = fitf.compute_drift_v(data.Z, nbins=nbins_dv,
                                    zrange=zrange_dv, detector=detector)

    ## energy resolution and error
    tot_corr_factor = apply_all_correction(maps = emaps,
                                           apply_temp=False)
    nbins = int((len(data.S2e))**0.5)
    f = fitf.quick_gauss_fit(data.S2e.values*tot_corr_factor(
                                  data.X.values,
                                  data.Y.values,
                                  data.Z.values,
                                  data.time.values),
                        bins=nbins)
    R = resolution(f.values, f.errors, 41.5)
    resol, err_resol = R[0][0], R[0][1]
    ## average values
    parameters = ['S1w', 'S1h', 'S1e',
                  'S2w', 'S2h', 'S2e', 'S2q',
                  'Nsipm', 'Xrms', 'Yrms']
    mean_d, var_d = {}, {}
    for parameter in parameters:
        data_value           = getattr(data, parameter)
        mean_d[parameter] = np.mean(data_value)
        var_d [parameter] = (np.var(data_value)/len(data_value))**0.5

    ## saving as DataFrame
    pars = DataFrame({'ts'   : [ts]             ,
                         'e0'   : [e0]             , 'e0u'   : [e0u]           ,
                         'lt'   : [lt]             , 'ltu'   : [ltu]           ,
                         'dv'   : [dv]             , 'dvu'   : [dvu]           ,
                         'resol': [resol]          , 'resolu': [err_resol]     ,
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

    return pars

def cut_time_evolution(masks_time : List[np.array],
                       dst        : DataFrame,
                       masks_cuts : MasksContainer,
                       pars_table : DataFrame):

    """
    Computes the efficiency evolution in time for a given krypton distribution
    for different time slices.
    Returns the input DataFrame updated with new 3 columns.
    Parameters
    ----------
    masks: list of boolean lists
        Allows dividing the distribution into time slices.
    data: DataFrame
        Kdst distribution to analyze.
    masks_cuts: MasksContainer
        Container for the S1, S2 and Band cuts masks.
        The masks don't have to be inclusive.
    pars: DataFrame
        Each column corresponds to the average value for a given parameter.
        Each row corresponds to the parameters for a given time slice.
    Returns
    -------
    parspars_table_out: DataFrame
        pars Table imput updated with 3 new columns, one for each cut.
    """

    len_ts = len(masks_time)
    n0     = np.zeros(len_ts)
    nS1    = np.zeros(len_ts)
    nS2    = np.zeros(len_ts)
    nBand  = np.zeros(len_ts)
    for index in range(len_ts):
        t_mask       = masks_time[index]
        n0   [index] = dst[t_mask].event.nunique()
        nS1mask      = t_mask  & masks_cuts.s1
        nS1  [index] = dst[nS1mask].event.nunique()
        nS2mask      = nS1mask & masks_cuts.s2
        nS2  [index] = dst[nS2mask].event.nunique()
        nBandmask    = nS2mask & masks_cuts.band
        nBand[index] = dst[nBandmask].event.nunique()

    pars_table_out = pars_table.assign(S1eff   = nS1   / n0,
                                       S2eff   = nS2   / nS1,
                                       Bandeff = nBand / nS2)
    return pars_table_out


def resolution(values, errors = None, E_from=41.5, E_to=2458):
    """
    Compute resolution at E_from and resolution at E_to
    with uncertainty propagation.
    """
    if errors is None:
        errors = np.zeros_like(values)

    amp  ,   mu,   sigma, *_ = values
    u_amp, u_mu, u_sigma, *_ = errors

    r   = 235. * sigma/mu
    u_r = r * (u_mu**2/mu**2 + u_sigma**2/sigma**2)**0.5

    scale = (E_from/E_to)**0.5
    return Measurement(r        , u_r        ), \
           Measurement(r * scale, u_r * scale)