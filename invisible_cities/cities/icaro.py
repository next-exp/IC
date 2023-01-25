"""
-----------------------------------------------------------------------
                                Icaro
-----------------------------------------------------------------------


"""
from operator import attrgetter

import tables as tb

from .. reco         import tbl_functions as tbl
from .. io.krmaps_io import write_krmap
from .. io.krmaps_io import write_krevol
from .. io.krmaps_io import write_mapinfo

from .. dataflow          import dataflow as fl
from .. dataflow.dataflow import push
from .. dataflow.dataflow import pipe

from .  components import city
from .  components import print_numberofevents
from .  components import dst_from_files
from .  components import quality_check
from .  components import kr_selection
from .  components import map_builder
from .  components import add_krevol


@city
def icaro(files_in, file_out, compression, detector_db, run_number,
          bootstrap, quality_ranges, band_sel_params, map_params, krevol_params):


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

    add_krevol_map       = fl.map(add_krevol(map_params.xybins, krevol_params),
                                  args = "kr_data", "kr_mask",
                                  out  = "evolution"           )


    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        # Define writers...
        write_krmap_sink      = fl.sink(  write_krmap(h5out), args=("run_number", "event_number", "timestamp"))
        write_krevol_sink     = fl.sink( write_krevol(h5out), args="pointlike_event")
        write_mapinfo_sink    = fl.sink(write_mapinfo(h5out), args=("event_number", "pmap_passed"))

        return push(source = dst_from_files(files_in, "DST", "Events"),
                    pipe   = pipe(
                        fl.slice(*event_range, close_all=True),
                        print_every(print_mod)                ,
                        print_numberofevents                  ,
                        quality_check_before                  ,
                        kr_selection_map                      ,
                        quality_check_after                   ,
                        print_numberofevents                  ,
                        map_builder                           ,
                        add_krevol                            ,
                        fl.fork(write_krmap                   ,
                                write_krevol                  ,
                                write_mapinfo                 )),
                    result = dict(events_in  = event_count_in .future,
                                  events_out = event_count_out.future,
                                  selection  = kr_selection   .future))
