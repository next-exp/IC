"""
-----------------------------------------------------------------------
                              Isaura
-----------------------------------------------------------------------

From ancient greek, Ισαυρία: an ancient rugged region in Asia Minor.

This city computes tracks from the deconvolved hits (dDST) and
extracts topology information. It produces the same output as Esmeralda.
"""

import tables as tb

from .. core.configure      import EventRangeType
from .. core.configure      import OneOrManyFiles
from .. core                import tbl_functions        as tbl
from .. dataflow            import dataflow             as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe

from .  components import city
from .  components import print_every
from .  components import collect
from .  components import copy_mc_info
from .  components import dhits_from_files
from .  components import compute_and_write_tracks_info

from .. types.symbols import HitEnergy

from ..  io.run_and_event_io import run_and_event_writer
from ..  io.         hits_io import hits_writer
from ..  io.         kdst_io import kdst_from_df_writer



@city
def isaura( files_in       : OneOrManyFiles
          , file_out       : str
          , compression    : str
          , event_range    : EventRangeType
          , print_mod      : int
          , detector_db    : str
          , run_number     : int
          , paolina_params : dict
          ):
    """
    The city extracts topology information.
    ----------
    Parameters
    ----------
    files_in  : str, filepath
         input file
    file_out  : str, filepath
         output file
    compression : str
         Default  'ZLIB4'
    event_range : int /'all_events'
         number of events from files_in to process
    print_mode : int
         how frequently to print events
    run_number : int
         has to be negative for MC runs
    paolina_params               :dict
        vox_size                 : [float, float, float]
            (maximum) size of voxels for track reconstruction
        strict_vox_size          : bool
            if False allows per event adaptive voxel size,
            smaller of equal thatn vox_size
        energy_threshold        : float
            if energy of end-point voxel is smaller
            the voxel will be dropped and energy redistributed to the neighbours
        min_voxels              : int
            after min_voxel number of voxels is reached no dropping will happen.
        blob_radius             : float
            radius of blob
        max_num_hits            : int
            maximum number of hits allowed per event to run paolina functions.
    ----------
    Input
    ----------
    Beersheba output
    ----------
    Output
    ----------
    - MC info (if run number <=0)
    - Tracking/Tracks - summary of per track information
    - Summary/events  - summary of per event information
"""

    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()

    filter_out_none = fl.filter(lambda x: x is not None, args = "kdst")

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:
        write_event_info = fl.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))
        write_kdst_table = fl.sink( kdst_from_df_writer(h5out), args= "kdst")

        write_hits = fl.sink(hits_writer(h5out, "DECO", "Events"), args="paolina_hits") # from within compute_tracks

        evtnum_collect = collect()

        compute_tracks = compute_and_write_tracks_info(paolina_params, h5out, hit_type=HitEnergy.E)

        result = push(source = dhits_from_files(files_in),
                      pipe   = pipe(fl.slice(*event_range, close_all=True)        ,
                                    print_every(print_mod)                        ,
                                    event_count_in        .spy                    ,
                                    fl.branch("event_number", evtnum_collect.sink),
                                    event_count_out       .spy                    ,
                                    fl.fork( compute_tracks
                                           , write_event_info
                                           , write_hits
                                           , (filter_out_none, write_kdst_table)) ),
                      result = dict(events_in  =event_count_in .future,
                                    events_out =event_count_out.future,
                                    evtnum_list=evtnum_collect .future))


        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)

        return result
