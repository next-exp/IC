import numpy  as np
import tables as tb
import pandas as pd

from .  components                       import city
from .  components                       import get_run_number
from .. dataflow                         import dataflow        as fl
from .. io      .dst_io                  import load_dsts

from .. core    .configure               import EventRangeType
from .. core    .configure               import OneOrManyFiles
from .. core    .system_of_units         import keV
from .. core                             import tbl_functions   as tbl

from .. types   .symbols                 import NormMethod
from .. types   .symbols                 import SelRegionMethod
from .. types   .symbols                 import MapFitFunction

from .. icaros  .correction_functions    import apply_correctionmap_inplace_kdst
from .. icaros  .selection_functions     import apply_selections
from .. icaros  .krmap_functions         import compute_3D_map
from .. icaros  .krmap_functions         import gaussian_fit_ready
from .. icaros  .krmap_functions         import get_median
from .. icaros  .krmap_functions         import compute_metadata
from .. icaros  .krmap_functions         import get_time_evol
from .. icaros  .krmap_functions         import save_map
from .. icaros  .control_plots_functions import make_control_plots
from .. icaros  .control_plots_functions import plot_time_evolution_with_errors_and_dates

from typing import Tuple
from typing import Union
from typing import Callable
from typing import List
from typing import Iterator
from typing import Dict


def concatenated_dsts_from_files(path: List[str], group: str, node:str)-> Iterator[Dict[str,Union[pd.DataFrame, int, np.ndarray]]]:
    df = load_dsts(path, group, node)
    with tb.open_file(path[0], 'r') as h5in:
        run_number = get_run_number(h5in)

    yield dict(dst = df,
               run_number = run_number
               )


def apply_map(pre_map, norm_method, xy_params, col_name, unit):
    pre_map = pd.read_hdf(pre_map)
    def apply_3Dmap(df):
        return apply_correctionmap_inplace_kdst(df, pre_map, norm_method, xy_params, col_name, unit)
    return apply_3Dmap


def select_dst(dtrms2_low, dtrms2_upp, low_xrays, high_xrays, low_S2t, high_S2t, R_max, low_DT, high_DT, low_nsipm, high_nsipm):
    def apply_selections_dst(df):
        return apply_selections(df, dtrms2_low, dtrms2_upp, low_xrays, high_xrays, low_S2t, high_S2t, R_max, low_DT, high_DT, low_nsipm, high_nsipm)
    return apply_selections_dst


def create_selfmap(xy_range, dt_range, xy_nbins, dt_nbins, S2e_range, fit_function, nbins, min_events):
    if fit_function == MapFitFunction.gaussian:
        fit_function = gaussian_fit_ready(nbins, min_events)

    elif fit_function == MapFitFunction.median:
        fit_function = get_median

    else:
        raise ValueError(f'Invalid fit function {fit_function}')

    def create_map(df):
        return compute_3D_map(df,xy_range, dt_range, xy_nbins, dt_nbins, S2e_range, fit_function)
    return create_map


def get_metadata(xy_range, dt_range, xy_nbins, dt_nbins):
    def metadata(df, krmap):
        return compute_metadata(df, krmap, xy_range, dt_range, xy_nbins, dt_nbins)
    return metadata


def apply_selfmap(norm_method, xy_params, col_name, unit):
    def apply_3Dselfmap(df, map3D):
        return apply_correctionmap_inplace_kdst(df, map3D, norm_method, xy_params, col_name, unit)
    return apply_3Dselfmap


def time_evol(slice_hours,  x0, y0, shape, shape_size, dtbins_dv, s1_DTrange, bins_Ec, error):
    def get_time_evolution(df, run_number):
        return get_time_evol(df, slice_hours, run_number, x0, y0, shape, shape_size, dtbins_dv, s1_DTrange, bins_Ec, error)
    return get_time_evolution


def save_krmap(name):
    def save(efficiencies, krmap, metadata, t_evol):
        return save_map(name, efficiencies, krmap, metadata, t_evol)
    return save


def do_control_plots(plots_out,ebins1, ns1bins, s1hbins, s1wbins, ebins2, ns2bins, s2hbins, s2qbins, qmaxbins, s2wbins, dtrms2_low, dtrms2_upp, drms2_cen, dtbins2, bins, dtrs2_bins, statistic, x0, y0, shape, shape_size, xy_range_plot):
    def control_plots(df, df_corr, efficiencies, run_number):
        return make_control_plots(df, df_corr, efficiencies, run_number, plots_out, ebins1, ns1bins, s1hbins, s1wbins, ebins2, ns2bins, s2hbins, s2qbins, qmaxbins, s2wbins, dtrms2_low, dtrms2_upp, drms2_cen,dtbins2, bins, dtrs2_bins, statistic, x0, y0, shape, shape_size, xy_range_plot)
    return control_plots


def time_evol_plots(plots_out):
    def t_evol_plots(t_evol):
        return plot_time_evolution_with_errors_and_dates(t_evol, plots_out)
    return t_evol_plots


@city
def zemrude(files_in           : OneOrManyFiles
            , file_out         : str
            , plots_out        : str
            , compression      : str
            , event_range      : EventRangeType
            , detector_db      : str
            , run_number       : int
            , pre_map          : str
            , norm_method      : NormMethod
            , dtrms2_low       : Callable
            , dtrms2_upp       : Callable
            , dtrms2_cen       : Callable
            , low_xrays        : float
            , high_xrays       : float
            , low_S2t          : float
            , high_S2t         : float
            , R_max            : float
            , low_DT           : float
            , high_DT          : float
            , low_nsipm        : int
            , high_nsipm       : int
            , xy_range         : tuple
            , dt_range         : tuple
            , xy_nbins         : int
            , dt_nbins         : int
            , S2e_range        : tuple
            , fit_function     : MapFitFunction
            , min_events       : int
            , nbins            : int
            , slice_hours      : float
            , x0               : float
            , y0               : float
            , shape            : SelRegionMethod
            , shape_size       : float
            , dtbins_dv        : np.ndarray
            , s1_DTrange       : tuple
            , bins_Ec          : np.ndarray
            , ebins1           : np.ndarray
            , ns1bins          : np.ndarray
            , s1hbins          : np.ndarray
            , s1wbins          : np.ndarray
            , ebins2           : np.ndarray
            , ns2bins          : np.ndarray
            , s2hbins          : np.ndarray
            , s2qbins          : np.ndarray
            , qmaxbins         : np.ndarray
            , s2wbins          : np.ndarray
            , dtbins2          : np.ndarray
            , bins             : int
            , dtr2_bins        : tuple
            , statistic        : str
            , xy_range_plot    : np.ndarray
            , error            : bool = False
            , xy_params        : dict = None
            ):

    apply_preliminary_map  = fl.map( apply_map(pre_map,
                                              norm_method,
                                              xy_params,
                                              'Ec',
                                              unit = keV)
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
                                            S2e_range,
                                            fit_function,
                                            nbins,
                                            min_events)
                             , args = 'selected_dst'
                             , out  = '3D_krmap')


    compute_metadata = fl.map( get_metadata(xy_range,
                                            dt_range,
                                            xy_nbins,
                                            dt_nbins)
                               , args = ('selected_dst', '3D_krmap')
                               , out = 'metadata')

    apply_3Dmap_to_data = fl.map( apply_selfmap(norm_method,
                                                xy_params,
                                                'Ec_2',
                                                 keV)
                                  , args = ('selected_dst', '3D_krmap')
                                  , out  =  'corrected_dst')


    get_t_evol = fl.map( time_evol(slice_hours,
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

    save_everything = fl.sink( save_krmap(file_out)
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
                                                  statistic,
                                                  x0,
                                                  y0,
                                                  shape,
                                                  shape_size,
                                                  xy_range_plot
                                                  )

                                 , args = ('dst', 'corrected_dst','efficiencies', 'run_number')
                                 )

    plot_t_evol = fl.sink(time_evol_plots(plots_out)
                          , args = ('time_evol'))

    #create/overwrite the output (write mode). The output file is created
    #with the compression settings from tbl_functions.
    with tb.open_file(file_out, "w", filters=tbl.filters(compression)):
        pass
    fl.push( source = concatenated_dsts_from_files(files_in, "DST", "Events")
            ,pipe   = fl.pipe(apply_preliminary_map,
                               apply_selections,
                               compute_3D_map,
                               compute_metadata,
                               apply_3Dmap_to_data,
                               get_t_evol,

                               fl.fork(save_everything,
                                       make_control_plots,
                                       plot_t_evol))
            )
