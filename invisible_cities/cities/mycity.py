import os
import numpy  as np
import tables as tb
import pandas as pd
from functools import partial

from .. dataflow                  import dataflow        as fl

from .. cities  .components       import city
from .. cities  .components       import dst_from_files
from .. core                      import system_of_units as units
from .. core    .core_functions   import in_range
from .. core    .configure        import EventRangeType
from .. core    .configure        import OneOrManyFiles
from .. core                      import tbl_functions as tbl
from .. database.load_db          import DataSiPM
from .. io      .dst_io           import df_writer, load_dsts
from .. io      .run_and_event_io import run_and_event_writer

from .. reco    .psf_functions    import create_psf
from .. reco    .psf_functions    import hdst_psf_processing
from .. icaros  .correction_functions import apply_correctionmap_inplace
from .. icaros  .selection_functions import apply_selections
from .. icaros  .krmap_functions   import compute_3D_map, gaussian_fit, get_median, compute_metadata, get_time_evol, save_map
from .. types   .symbols          import NormMethod, SelRegionMethod, MapFitFunction
from .. icaros  .control_plots_functions import make_control_plots


from pathlib import Path
from typing import Sequence
from typing import Optional
from typing import Tuple
from typing import Union
from .components import *



def concatenated_dsts_from_files(path: List[str], group: str, node:str)-> Iterator[Dict[str,Union[pd.DataFrame, int, np.ndarray]]]:

    df = load_dsts(path, group, node)
    with tb.open_file(path[0], 'r') as h5in:
        run_number = get_run_number(h5in)

    yield dict(dst = df,
               run_number = run_number
               )


def apply_map(pre_map, norm_method, xy_params, col_name, keV):
    pre_map = pd.read_hdf(pre_map)
    def apply_3Dmap(df):
        return apply_correctionmap_inplace(df, pre_map, norm_method, xy_params, col_name, keV)
    return apply_3Dmap



def select_dst(dtrms2_low, dtrms2_upp, low_xrays, high_xrays, low_S2t, high_S2t, R_max, low_DT, high_DT, low_nsipm, high_nsipm):
    def apply_selections_dst(df):
        return apply_selections(df, dtrms2_low, dtrms2_upp, low_xrays, high_xrays, low_S2t, high_S2t, R_max, low_DT, high_DT, low_nsipm, high_nsipm)
    return apply_selections_dst



def create_selfmap(xy_range, dt_range, xy_nbins, dt_nbins, fit_function, nbins_S2e, S2e_range):
    if fit_function == MapFitFunction.gaussian_fit:
        fit_function = gaussian_fit
    else:
        fit_function = get_median
    def create_map(df):
        return compute_3D_map(df,xy_range, dt_range, xy_nbins, dt_nbins, fit_function, nbins_S2e, S2e_range)
    return create_map



def get_metadata(xy_range, dt_range, xy_nbins, dt_nbins):
    def metadata(df, krmap):
        return compute_metadata(df, krmap, xy_range, dt_range, xy_nbins, dt_nbins)
    return metadata



def apply_selfmap(norm_method, xy_params, col_name, keV):
    def apply_3Dselfmap(df, map3D):
        return apply_correctionmap_inplace(df, map3D, norm_method, xy_params, col_name, keV)
    return apply_3Dselfmap


def time_evol(slice_hours, col_name1, col_name2,  x0, y0, shape, shape_size, dtbins_dv, s1_DTrange, bins_Ec, error):
    def get_time_evolution(df, run_number):
        return get_time_evol(df, slice_hours, col_name1,col_name2, run_number, x0, y0, shape, shape_size, dtbins_dv, s1_DTrange, bins_Ec, error)
    return get_time_evolution

def save_krmap(name):
    def save(efficiencies, krmap, metadata, t_evol):
        return save_map(name, efficiencies, krmap, metadata, t_evol)
    return save

def do_control_plots(plots_out,ebins1, ns1bins, s1hbins, s1wbins, ebins2, ns2bins, s2hbins, s2qbins, qmaxbins, s2wbins, dtrms2_low, dtrms2_upp, drms2_cen, dtbins2, bins, dtrs2_bins, col_name1, col_name2,statistic, x0, y0, shape, shape_size, xy_range_plot):
    #os.mkdir(plots_out)
    def control_plots(df, df_sel, df_corr, run_number):
        return make_control_plots(df, df_sel, df_corr, run_number, plots_out, ebins1, ns1bins, s1hbins, s1wbins, ebins2, ns2bins, s2hbins, s2qbins, qmaxbins, s2wbins, dtrms2_low, dtrms2_upp, drms2_cen,dtbins2, bins, dtrs2_bins, col_name1, col_name2, statistic, x0, y0, shape, shape_size, xy_range_plot)
    return control_plots





@city
def mycity(files_in    : OneOrManyFiles
            , file_out    : str
            , plots_out   : str
            , compression : str
            , event_range : EventRangeType
            , detector_db : str
            , run_number  : int
            , pre_map     : str
            , norm_method : NormMethod
            , dtrms2_low  : Callable
            , dtrms2_upp  : Callable
            , dtrms2_cen  : Callable
            , low_xrays   : float
            , high_xrays  : float
            , low_S2t     : float
            , high_S2t    : float
            , R_max       : float
            , low_DT      : float
            , high_DT     : float
            , low_nsipm   : int
            , high_nsipm  : int
            , xy_range    : tuple
            , dt_range    : tuple
            , xy_nbins    : int
            , dt_nbins    : int
            , fit_function: MapFitFunction
            , nbins_S2e   : int
            , S2e_range   : tuple
            , slice_hours : float
            , x0          : float
            , y0          : float
            , shape       : SelRegionMethod
            , shape_size  : float
            , dtbins_dv   : np.ndarray
            , s1_DTrange  : tuple
            , bins_Ec     : np.ndarray
            , name        : str
            , ebins1      : np.ndarray
            , ns1bins     : np.ndarray
            , s1hbins     : np.ndarray
            , s1wbins     : np.ndarray
            , ebins2      : np.ndarray
            , ns2bins     : np.ndarray
            , s2hbins     : np.ndarray
            , s2qbins     : np.ndarray
            , qmaxbins    : np.ndarray
            , s2wbins     : np.ndarray
            , dtbins2     : np.ndarray
            , bins        : int
            , dtr2_bins   : tuple
            , col_name1   : str
            , col_name2   : str
            , statistic   : str
            , xy_range_plot    : np.ndarray
            , error       : bool = False
            , xy_params   : dict = None
            ):



    apply_preliminary_map = fl.map( apply_map(pre_map,
                                              norm_method,
                                              xy_params,
                                              'Ec',
                                              keV=True)
                                  , item = 'dst')



    apply_selections = fl.map( select_dst(dtrms2_low,
                                          dtrms2_upp,
                                          low_xrays,
                                          high_xrays,
                                          low_S2t,
                                          high_S2t,
                                          R_max,
                                          low_DT,
                                          high_DT,
                                          low_nsipm,
                                          high_nsipm)
                              , args = 'dst'
                              , out = ('selected_dst', 'efficiencies'))


    compute_3D_map = fl.map( create_selfmap(xy_range,
                                            dt_range,
                                            xy_nbins,
                                            dt_nbins,
                                            fit_function,
                                            nbins_S2e,
                                            S2e_range)
                             , args = 'selected_dst'
                             , out = '3D_krmap')


    compute_metadata = fl.map( get_metadata(xy_range,
                                            dt_range,
                                            xy_nbins,
                                            dt_nbins)
                               , args = ('selected_dst', '3D_krmap')
                               , out = 'metadata')

    apply_3Dmap_to_data = fl.map( apply_selfmap(norm_method,
                                                xy_params,
                                                'Ec_2',
                                                 True)
                                  , args = ('selected_dst', '3D_krmap')
                                  , out = 'corrected_dst')


    get_t_evol = fl.map( time_evol(slice_hours,
                                    'Ec',
                                    'Ec_2',
                                    x0,
                                    y0,
                                    shape,
                                    shape_size,
                                    dtbins_dv,
                                    s1_DTrange,
                                    bins_Ec,
                                    error
                                    )
                         , args = ('corrected_dst', 'run_number')
                         , out = 'time_evol')

    save_everything = fl.sink( save_krmap(name)
                             , args = ('efficiencies', '3D_krmap', 'metadata', 'time_evol'))


    make_control_plots = fl.sink(do_control_plots(plots_out,
                                                 ebins1,
                                                 ns1bins,
                                                 s1hbins,
                                                 s1wbins,
                                                 ebins2,
                                                 ns2bins,
                                                 s2hbins,
                                                 s2qbins,
                                                 qmaxbins,
                                                 s2wbins,
                                                 dtrms2_low,
                                                 dtrms2_upp,
                                                 dtrms2_cen,
                                                 dtbins2,
                                                 bins,
                                                 dtr2_bins,
                                                 col_name1,
                                                 col_name2,
                                                 statistic,
                                                 x0,
                                                 y0,
                                                 shape,
                                                 shape_size,
                                                 xy_range_plot
                                                 )
                                 , args = ('dst', 'selected_dst', 'corrected_dst', 'run_number')
                                 )


    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:
        result = fl.push( source = concatenated_dsts_from_files(files_in, "DST", "Events")
                        , pipe   = fl.pipe(apply_preliminary_map,
                                           apply_selections,
                                           compute_3D_map,
                                           compute_metadata,
                                           apply_3Dmap_to_data,
                                           get_t_evol,

                                           fl.fork(save_everything,
                                                   make_control_plots))
#                        , result = None
                        )
