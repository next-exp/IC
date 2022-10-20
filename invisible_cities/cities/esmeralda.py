"""
-----------------------------------------------------------------------
                              Esmeralda
-----------------------------------------------------------------------
From Spanish esmeralda (“emerald”), as first used in the novel Notre-Dame de Paris (1831) by Victor Hugo.
This city corrects hits energy and extracts topology information.
The input is penthesilea output containing hits, kdst global information and mc info.
The city outputs :
    - CHITS corrected hits table,
        - lowTh  - contains corrected hits table that passed h.Q >= charge_threshold_low constrain
        - highTh - contains corrected hits table that passed h.Q >= charge_threshold_high constrain.
                   it also contains:
                   - Ep field that is the energy of a hit after applying drop_end_point_voxel algorithm.
                   - track_id denoting to which track from Tracking/Tracks dataframe the hit belong to
    - MC info (if run number <=0)
    - Tracking/Tracks - summary of per track information
    - Summary/events  - summary of per event information
    - DST/Events      - copy of kdst information from penthesilea
"""

import tables as tb

from .. core.configure      import EventRangeType
from .. core.configure      import OneOrManyFiles
from .. reco                import tbl_functions        as tbl
from .. evm                 import event_model          as evm
from .. dataflow            import dataflow             as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe

from .  components import city
from .  components import print_every
from .  components import collect
from .  components import copy_mc_info
from .  components import hits_and_kdst_from_files
from .  components import hits_thresholder
from .  components import compute_and_write_tracks_info

from .. io.         hits_io import hits_writer
from .. io.         kdst_io import kdst_from_df_writer
from .. io.run_and_event_io import run_and_event_writer


def hit_dropper(radius : float):
    def in_fiducial(hit : evm.Hit) -> bool:
        return hit.R < radius

    def drop_hits(hitc : evm.HitCollection) -> evm.HitCollection:
        hitc.hits = list(filter(in_fiducial, hitc.hits))
        return hitc

    return drop_hits


@city
def esmeralda( files_in       : OneOrManyFiles
             , file_out       : str
             , compression    : str
             , event_range    : EventRangeType
             , print_mod      : int
             , detector_db    : str
             , run_number     : int
             , threshold      : float
             , same_peak      : bool
             , fiducial_r     : float
             , paolina_params : dict
             ):
    """
    The city applies a threshold to sipm hits and extracts
    topology information.

    Parameters
    ----------
    files_in  : str, filepath
         input file
    file_out  : str, filepath
         output file
    compression : str
         Default  'ZLIB4'
    event_range : EventRangeType
         number of events from files_in to process
    print_mode : int
         how frequently to print events
    run_number : int
         has to be negative for MC runs
    threshold : float
        minimum pes for a hit (energy)
    same_peak : bool
        whether to reassign NN hits' energy only to the hits from the same peak

    paolina_params              :dict
        vox_size                : [float, detfloat, float]
            (maximum) size of voxels for track reconstruction
        strict_vox_size         : bool
            if False allows per event adaptive voxel size,
            smaller of equal thatn vox_size from sophronia
        energy_threshold        : float
            if energy of end-point voxel is smaller
            the voxel will be dropped and energy redistributed to the neighbours
        min_voxels              : int
            after min_voxel number of voxels is reached no dropping will happen.
        blob_radius             : float
            radius of blob
        max_num_hits            : int
            maximum number of hits allowed per event to run paolina functions.

    Input
    ----------
    /RECO/Events
    /DST/Events

    Output
    ----------
    - CHITS corrected hits table,
        - highTh - contains corrected hits table that passed h.Q >= charge_threshold_high constrain.
                   it also contains:
                   - Ep field that is the energy of a hit after applying drop_end_point_voxel algorithm.
                   - track_id denoting to which track from Tracking/Tracks dataframe the hit belong to
    - MC info (if run number <=0)
    - Tracking/Tracks - summary of per track information
    - Summary/events  - summary of per event information
    - DST/Events      - kdst information
    """

    drop_external_hits = fl.map(hit_dropper(fiducial_r), item="hits")
    threshold_hits  = fl.map(hits_thresholder(threshold, same_peak), item="hits")
    event_count_in  = fl.spy_count()
    event_count_out = fl.count()

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:

        write_event_info   = fl.sink( run_and_event_writer(h5out)
                                    , args = "run_number event_number timestamp".split())

        write_paolina_hits = fl.sink(hits_writer( h5out
                                                , group_name = "CHITS"
                                                , table_name = "highTh")
                                    , args = "Ep_hits")

        write_kdst         = fl.sink( kdst_from_df_writer(h5out)
                                    , args = "kdst")

        compute_tracks = compute_and_write_tracks_info( paolina_params
                                                      , h5out
                                                      , evm.HitEnergy.Ec
                                                      , "high_th_select"
                                                      , write_paolina_hits)

        event_number_collector = collect()

        collect_evts = "event_number", fl.fork( event_number_collector.sink
                                              , event_count_out       .sink)

        result = push(source = hits_and_kdst_from_files(files_in, "RECO", "Events"),
                      pipe   = pipe( fl.slice(*event_range, close_all=True)
                                   , print_every(print_mod)
                                   , event_count_in.spy
                                   , drop_external_hits
                                   , threshold_hits
                                   , compute_tracks
                                   , fl.fork( write_kdst
                                            , write_event_info
                                            , collect_evts
                                            )),

                      result = dict(events_in   = event_count_in        .future,
                                    events_out  = event_count_out       .future,
                                    evtnum_list = event_number_collector.future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)

        return result
