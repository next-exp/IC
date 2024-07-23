"""
-----------------------------------------------------------------------
                              Penthesilea
-----------------------------------------------------------------------

From ancient Greek, Πενθεσίλεια: She, who brings suffering.

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
from operator import attrgetter

import tables as tb

from .. core.configure         import       EventRangeType
from .. core.configure         import       OneOrManyFiles
from .. core                   import       tbl_functions as tbl
from .. io  .          hits_io import          hits_writer
from .. io  . run_and_event_io import run_and_event_writer
from .. io  .          kdst_io import            kr_writer
from .. io  .  event_filter_io import  event_filter_writer

from .. dataflow          import dataflow as df
from .. dataflow.dataflow import     push
from .. dataflow.dataflow import     pipe
from .. types.symbols     import RebinMethod
from .. types.symbols     import  SiPMCharge
from .. types.symbols     import      XYReco

from .  components import                  city
from .  components import          copy_mc_info
from .  components import           print_every
from .  components import       peak_classifier
from .  components import   compute_xy_position
from .  components import       pmap_from_files
from .  components import           hit_builder
from .  components import               collect
from .  components import build_pointlike_event as build_pointlike_event_


@city
def penthesilea( files_in           : OneOrManyFiles
               , file_out           : str
               , compression        : str
               , event_range        : EventRangeType
               , print_mod          : int
               , detector_db        : str
               , run_number         : int
               , drift_v            : float
               , rebin              : int
               , s1_nmin            :   int, s1_nmax     :   int
               , s1_emin            : float, s1_emax     : float
               , s1_wmin            : float, s1_wmax     : float
               , s1_hmin            : float, s1_hmax     : float
               , s1_ethr            : float
               , s2_nmin            :   int, s2_nmax     :   int
               , s2_emin            : float, s2_emax     : float
               , s2_wmin            : float, s2_wmax     : float
               , s2_hmin            : float, s2_hmax     : float
               , s2_ethr            : float
               , s2_nsipmmin        :   int, s2_nsipmmax :   int
               , slice_reco_algo    : XYReco
               , global_reco_algo   : XYReco
               , slice_reco_params  : dict
               , global_reco_params : dict
               , rebin_method       : RebinMethod
               , sipm_charge_type   : SiPMCharge
               ):

    #  slice_reco_params are qth, qlm, lm_radius, new_lm_radius, msipm used for hits reconstruction
    # global_reco_params are qth, qlm, lm_radius, new_lm_radius, msipm used for overall global (pointlike event) reconstruction
    slice_reco  = compute_xy_position(detector_db, run_number,  slice_reco_algo, ** slice_reco_params)
    global_reco = compute_xy_position(detector_db, run_number, global_reco_algo, **global_reco_params)


    classify_peaks = df.map(peak_classifier(**locals()),
                            args = "pmap",
                            out  = "selector_output")

    pmap_passed           = df.map(attrgetter("passed"), args="selector_output", out="pmap_passed")
    pmap_select           = df.count_filter(bool, args="pmap_passed")

    build_hits            = df.map(hit_builder(detector_db, run_number, drift_v,
                                               rebin, rebin_method,
                                               global_reco, slice_reco,
                                               sipm_charge_type),
                                   args = ("pmap", "selector_output", "event_number", "timestamp"),
                                   out  = "hits"                                                 )
    build_pointlike_event = df.map(build_pointlike_event_( detector_db, run_number, drift_v
                                                         , global_reco, sipm_charge_type),
                                   args = ("pmap", "selector_output", "event_number", "timestamp"),
                                   out  = "pointlike_event"                                      )

    event_count_in  = df.spy_count()
    event_count_out = df.spy_count()

    evtnum_collect = collect()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        # Define writers...
        write_event_info      = df.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))
        write_hits            = df.sink(         hits_writer(h5out), args="hits")
        write_pointlike_event = df.sink(           kr_writer(h5out), args="pointlike_event")
        write_pmap_filter     = df.sink( event_filter_writer(h5out, "s12_selector"), args=("event_number", "pmap_passed"))

        result = push(source = pmap_from_files(files_in),
                      pipe   = pipe(df.slice(*event_range, close_all=True)                ,
                                    print_every(print_mod)                                ,
                                    event_count_in.spy                                    ,
                                    classify_peaks                                        ,
                                    pmap_passed                                           ,
                                    df.branch(write_pmap_filter)                          ,
                                    pmap_select          .filter                          ,
                                    event_count_out      .spy                             ,
                                    df.branch("event_number", evtnum_collect.sink)        ,
                                    df.fork((build_hits           , write_hits           ),
                                            (build_pointlike_event, write_pointlike_event),
                                                                    write_event_info    )),
                      result = dict(events_in   = event_count_in .future,
                                    events_out  = event_count_out.future,
                                    evtnum_list = evtnum_collect .future,
                                    selection   = pmap_select    .future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)

        return result
