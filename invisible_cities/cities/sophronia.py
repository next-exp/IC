"""
-----------------------------------------------------------------------
                              Sophronia
-----------------------------------------------------------------------

From ancient Greek: sensible, prudent.

This city processes each S2 signal previously selected as pmaps in
irene assuming a unique S1 within an event to produce a set of
reconstructed energy depositions (hits). Hits consist of three
dimensional coordinates with associated energy (PMT signal) and charge
(SiPM signal). The city contains a peak/event filter, which can be
configured to find events with a certain number of S1/S2 signals that
satisfy certain properties. Currently, the city is designed to accept
only 1 S1 signal and will take the first S1 signal even if the filter
is configured to take more than 1 S1. Besides hits, the city also
stores the global (x, y) position of each S2 signal.
The tasks performed are:
    - Classify peaks according to the filter.
    - Filter out events that do not satisfy the selector conditions.
    - Rebin S2 signals.
    - Compute a set of hits for each slice in the rebinned S2 signal.
    - If there are more than one hit per slice, share the energy
      according to the charge recorded in the tracking plane.
"""

import os

from operator import attrgetter

import tables as tb

from .. core.configure         import       EventRangeType
from .. core.configure         import       OneOrManyFiles
from .. core.configure         import    check_annotations
from .. evm .event_model       import        HitCollection
from .. reco                   import        tbl_functions as tbl
from .. reco.corrections       import            read_maps
from .. reco.corrections       import apply_all_correction
from .. reco.corrections       import get_df_to_z_converter
from .. io  .          hits_io import          hits_writer
from .. io  . run_and_event_io import run_and_event_writer
from .. io  .          kdst_io import            kr_writer
from .. io  .  event_filter_io import  event_filter_writer

from .. dataflow          import     dataflow as df
from .. types.ic_types    import           NN
from .. types.symbols     import  RebinMethod
from .. types.symbols     import   SiPMCharge
from .. types.symbols     import       XYReco
from .. types.symbols     import NormStrategy

from .  components import                  city
from .  components import          copy_mc_info
from .  components import           print_every
from .  components import       peak_classifier
from .  components import   compute_xy_position
from .  components import       pmap_from_files
from .  components import         sipms_as_hits
from .  components import           hits_merger
from .  components import               collect
from .  components import build_pointlike_event as pointlike_event_builder

from typing import Callable


@check_annotations
def hits_corrector(map_fname : str, apply_temp : bool) -> Callable:
    """
    Applies energy correction map and converts drift time to z.

    Parameters
    ----------
    map_fname  : string (filepath)
        filename of the map
    apply_temp : bool
        whether to apply temporal corrections
        must be set to False if no temporal correction dataframe exists in map file

    Returns
    ----------
    A function that takes a HitCollection as input and returns
    the same object with modified Ec and Z fields.
    """
    map_fname = os.path.expandvars(map_fname)
    maps      = read_maps(map_fname)
    get_coef  = apply_all_correction(maps, apply_temp = apply_temp, norm_strat = NormStrategy.kr)
    time_to_Z = (get_df_to_z_converter(maps) if maps.t_evol is not None else
                 lambda x: x)

    def correct(hitc : HitCollection) -> HitCollection:
        for hit in hitc.hits:
            corr    = get_coef([hit.X], [hit.Y], [hit.Z], hitc.time)[0]
            hit.Ec  = hit.E * corr
            hit.xyz = (hit.X, hit.Y, time_to_Z(hit.Z)) # ugly, but temporary
        return hitc

    return correct


@check_annotations
def count_valid_hits(hitc : HitCollection):
    return sum(1 for hit in hitc.hits if hit.Q != NN)


@city
def sophronia( files_in           : OneOrManyFiles
             , file_out           : str
             , compression        : str
             , event_range        : EventRangeType
             , print_mod          : int
             , detector_db        : str
             , run_number         : int
             , drift_v            : float
             , s1_params          : dict
             , s2_params          : dict
             , global_reco_algo   : XYReco
             , global_reco_params : dict
             , rebin              : int
             , rebin_method       : RebinMethod
             , q_thr              : float
             , sipm_charge_type   : SiPMCharge
             , same_peak          : bool
             , corrections_file   : str
             , apply_temp         : bool
             ):

    global_reco = compute_xy_position( detector_db
                                     , run_number
                                     , global_reco_algo
                                     , **global_reco_params)

    classify_peaks = df.map( peak_classifier(**s1_params, **s2_params)
                           , args = "pmap"
                           , out  = "selector_output")

    pmap_passed    = df.map( attrgetter("passed")
                           , args = "selector_output"
                           , out  = "pmap_passed")

    pmap_select    = df.count_filter( bool
                                    , args = "pmap_passed")

    make_hits      = df.map( sipms_as_hits( detector_db
                                          , run_number
                                          , drift_v
                                          , rebin
                                          , rebin_method
                                          , q_thr
                                          , global_reco
                                          , sipm_charge_type)
                           , args = "pmap selector_output event_number timestamp".split()
                           , out  = "hits")

    enough_valid_hits = df.map( lambda hits: count_valid_hits(hits) > 0
                              , args = "hits"
                              , out  = "enough_valid_hits")

    hits_select    = df.count_filter( bool
                                    , args = "enough_valid_hits")

    merge_nn_hits  = df.map( hits_merger(same_peak)
                           , item = "hits")

    correct_hits   = df.map( hits_corrector(corrections_file, apply_temp)
                           , item = "hits")

    build_pointlike_event = df.map( pointlike_event_builder( detector_db
                                                           , run_number
                                                           , drift_v
                                                           , global_reco
                                                           , sipm_charge_type)
                                  , args = "pmap selector_output event_number timestamp".split()
                                  , out  = "pointlike_event")

    event_count_in         = df.spy_count()
    event_count_out        = df.spy_count()
    event_number_collector = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:
        write_event_info      = df.sink( run_and_event_writer(h5out)
                                       , args = "run_number event_number timestamp".split())
        write_hits            = df.sink(          hits_writer(h5out), args="hits")
        write_pointlike_event = df.sink(            kr_writer(h5out), args="pointlike_event")
        write_pmap_filter     = df.sink(  event_filter_writer(h5out, "s12_selector")
                                       , args = "event_number pmap_passed".split())
        write_hits_filter     = df.sink(  event_filter_writer(h5out, "valid_hit")
                                       , args = "event_number enough_valid_hits".split())

        hits_branch         = make_hits, merge_nn_hits, correct_hits, write_hits
        kdst_branch         = build_pointlike_event, write_pointlike_event
        collect_evt_numbers = "event_number", event_number_collector.sink

        result = df.push(source = pmap_from_files(files_in),

                         pipe   = df.pipe( df.slice(*event_range, close_all=True)
                                         , print_every(print_mod)
                                         , event_count_in.spy
                                         , classify_peaks
                                         , pmap_passed
                                         , df.branch(write_pmap_filter)
                                         , pmap_select    .filter
                                         , event_count_out.spy
                                         , make_hits
                                         , enough_valid_hits
                                         , df.branch(write_hits_filter)
                                         , hits_select.filter
                                         , df.fork( kdst_branch
                                                  , hits_branch
                                                  , collect_evt_numbers
                                                  , write_event_info)),

                         result = dict(events_in   = event_count_in        .future,
                                       events_out  = event_count_out       .future,
                                       evtnum_list = event_number_collector.future,
                                       selection   = pmap_select           .future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)

        return result
