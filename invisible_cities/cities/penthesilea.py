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
from .. core.system_of_units_c import units
from .. reco                   import tbl_functions        as tbl
from .. io  .         hits_io  import          hits_writer
from .. io  .       mcinfo_io  import       mc_info_writer
from .. io  .run_and_event_io  import run_and_event_writer
from .. io.           kdst_io  import            kr_writer
from .. io.   event_filter_io  import  event_filter_writer

from .. dataflow            import dataflow as df
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe

from .  components import city
from .  components import print_every
from .  components import peak_classifier
from .  components import compute_xy_position
from .  components import pmap_from_files
from .  components import hit_builder
from .  components import build_pointlike_event  as build_pointlike_event_

@city
def penthesilea(files_in, file_out, compression, event_range, print_mod, run_number,
                drift_v, rebin,
                s1_nmin, s1_nmax, s1_emin, s1_emax, s1_wmin, s1_wmax, s1_hmin, s1_hmax, s1_ethr,
                s2_nmin, s2_nmax, s2_emin, s2_emax, s2_wmin, s2_wmax, s2_hmin, s2_hmax, s2_ethr, s2_nsipmmin, s2_nsipmmax,
                slice_corona_reco_params        = dict(),
                slice_1hit_per_sipm_reco_params = dict(),
                global_reco_params              = dict()):
    #  slice_corona_reco_params are qth, qlm, lm_radius, new_lm_radius, msipm used for hits reconstruction using corona
    #  slice_1hit_per_sipm_reco_params are qth, qlm=0, lm_radius=0, new_lm_radius=0, msipm=1 used for hits reconstruction using corona ensuring 1 hit per sipm
    #  global_reco_params are qth, qlm, lm_radius, new_lm_radius, msipm used for overall global (pointlike event) reconstruction
    fixed_params_dict= dict(
        Qlm                    =  0 * units.pes,
        lm_radius              =  0 * units.mm ,
        new_lm_radius          =  0 * units.mm ,
        msipm                  =  1            )
    slice_1hit_per_sipm_reco_params.update(fixed_params_dict)

    classify_peaks                = df.map(peak_classifier(**locals()),
                                           args = "pmap",
                                           out  = "selector_output")
    
    pmap_passed                   = df.map(attrgetter("passed"), args="selector_output", out="pmap_passed")
    pmap_select                   = df.count_filter(bool, args="pmap_passed")
    
    reco_algo_slice_corona        = compute_xy_position(**slice_corona_reco_params)
    build_corona_hits             = df.map(hit_builder(run_number, drift_v, reco_algo_slice_corona, rebin),
                                           args = ("pmap", "selector_output", "event_number", "timestamp"),
                                           out  = "corona_hits"                                          )

    
    reco_algo_slice_1hit_per_sipm = compute_xy_position(**slice_1hit_per_sipm_reco_params)
    build_1hit_per_sipm_hits      = df.map(hit_builder(run_number, drift_v, reco_algo_slice_1hit_per_sipm, rebin),
                                           args = ("pmap", "selector_output", "event_number", "timestamp"       ),
                                           out  = "1hit_per_sipm_hits"                                          )
    

    reco_algo_global              = compute_xy_position(**global_reco_params)
    build_pointlike_event         = df.map(build_pointlike_event_(run_number, drift_v, reco_algo_global  ),
                                           args = ("pmap", "selector_output", "event_number", "timestamp"),
                                           out  = "pointlike_event"                                      )

    event_count_in  = df.spy_count()
    event_count_out = df.spy_count()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        # Define writers...
        write_mc_             = mc_info_writer(h5out) if run_number <= 0 else (lambda *_: None)

        write_mc                     = df.sink(write_mc_                                        , args=(        "mc", "event_number"             ))
        write_event_info             = df.sink(run_and_event_writer(h5out)                      , args=("run_number", "event_number", "timestamp"))
        write_corona_hits            = df.sink(         hits_writer(h5out, group_name = 'RECO'              ), args="corona_hits"                 )
        write_1hit_per_sipm_hits     = df.sink(         hits_writer(h5out, group_name = 'RECO_1HIT_PER_SIPM'), args="1hit_per_sipm_hits"          )
        write_pointlike_event        = df.sink(           kr_writer(h5out)                      , args="pointlike_event"                          )
        write_pmap_filter            = df.sink( event_filter_writer(h5out, "s12_selector")      , args=("event_number", "pmap_passed"            ))

        return push(source = pmap_from_files(files_in),
                    pipe   = pipe(
                        df.slice(*event_range, close_all=True),
                        print_every(print_mod)                ,
                        event_count_in       .spy             ,
                        classify_peaks                        ,
                        pmap_passed                           ,
                        df.branch(write_pmap_filter)          ,
                        pmap_select          .filter          ,
                        event_count_out      .spy             ,
                        df.fork((build_corona_hits            , write_corona_hits        ),
                                (build_1hit_per_sipm_hits     , write_1hit_per_sipm_hits ),
                                (build_pointlike_event        , write_pointlike_event    ),
                                                                write_mc                  ,
                                                                write_event_info         )),
                    result = dict(events_in  = event_count_in .future,
                                  events_out = event_count_out.future,
                                  selection  = pmap_select    .future))
