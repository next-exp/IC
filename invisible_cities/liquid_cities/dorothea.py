from operator import attrgetter

import tables as tb

from .. core.system_of_units_c import units
from .. reco                   import tbl_functions as tbl
from .. io.         kdst_io    import            kr_writer
from .. io.run_and_event_io    import run_and_event_writer

from .. dataflow          import dataflow      as fl
from .. dataflow.dataflow import push
from .. dataflow.dataflow import pipe

from .  components import city
from .  components import print_every
from .  components import pmap_from_files
from .  components import peak_classifier
from .  components import compute_xy_position
from .  components import build_pointlike_event  as build_pointlike_event_


@city
def dorothea(files_in, file_out, compression, event_range, print_mod, run_number,
             drift_v,
             s1_nmin, s1_nmax, s1_emin, s1_emax, s1_wmin, s1_wmax, s1_hmin, s1_hmax, s1_ethr,
             s2_nmin, s2_nmax, s2_emin, s2_emax, s2_wmin, s2_wmax, s2_hmin, s2_hmax, s2_ethr, s2_nsipmmin, s2_nsipmmax,
             qthr, qlm=0 * units.pes, lm_radius=-1 * units.mm, new_lm_radius=-1 * units.mm, msipm=1):
    # qlm           =  0 * pes every Cluster must contain at least one SiPM with charge >= qlm
    # lm_radius     = -1 * mm  by default, use overall barycenter for KrCity
    # new_lm_radius = -1 * mm  find a new cluster by calling barycenter() on pos/qs of SiPMs within
    #                          new_lm_radius of new_local_maximum
    # msipm         =  1       minimum number of SiPMs in a Cluster

    classify_peaks        = fl.map(peak_classifier(**locals()),
                                   args = "pmap",
                                   out  = "selector_output")

    pmap_select           = fl.count_filter(attrgetter("passed"), args="selector_output")

    reco_algo             = compute_xy_position(qthr, qlm, lm_radius, new_lm_radius, msipm)
    build_pointlike_event = fl.map(build_pointlike_event_(run_number, drift_v, reco_algo),
                                   args = ("pmap", "selector_output", "event_number", "timestamp"),
                                   out  = "pointlike_event"                                       )

    event_count_in        = fl.spy_count()
    event_count_out       = fl.spy_count()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        # Define writers...
        write_event_info      = fl.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))
        write_pointlike_event = fl.sink(           kr_writer(h5out), args="pointlike_event")

        return push(source = pmap_from_files(files_in),
                    pipe   = pipe(
                        fl.slice(*event_range, close_all=True),
                        print_every(print_mod)                ,
                        event_count_in       .spy             ,
                        classify_peaks                        ,
                        pmap_select          .filter          ,
                        event_count_out      .spy             ,
                        build_pointlike_event                 ,
                        fl.fork(write_pointlike_event         ,
                                write_event_info              )),
                    result = dict(events_in  = event_count_in .future,
                                  events_out = event_count_out.future,
                                  selection  = pmap_select    .future))
