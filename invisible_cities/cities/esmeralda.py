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

import os
import tables as tb
import numpy  as np

from typing      import Callable

from .. reco                import tbl_functions        as tbl
from .. reco                import hits_functions       as hif
from .. reco                import corrections          as cof
from .. evm                 import event_model          as evm
from .. dataflow            import dataflow             as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe

from .  components import city
from .  components import print_every
from .  components import collect
from .  components import copy_mc_info
from .  components import hits_and_kdst_from_files
from .  components import compute_and_write_tracks_info

from .. types.      ic_types import xy

from .. io.         hits_io import hits_writer
from .. io.         kdst_io import kdst_from_df_writer
from .. io.run_and_event_io import run_and_event_writer
from .. io. event_filter_io import event_filter_writer


def hits_threshold_and_corrector(map_fname        : str  ,
                                 threshold_charge : float,
                                 same_peak        : bool ,
                                 apply_temp       : bool
                                 ) -> Callable:
    """
    For a given correction map and hit threshold/ merging parameters returns a function that applies thresholding, merging and
    energy and Z corrections to a given HitCollection object.

    Parameters
    ----------
    map_fname        : string (filepath)
        filename of the map
    threshold_charge : float
        minimum pes of a hit
    same_peak        : bool
        if True energy of NN hits is assigned only to the hits from the same peak
    apply_temp       : bool
        whether to apply temporal corrections
        must be set to False if no temporal correction dataframe exists in map file
    norm_strat       :  norm_strategy
    class norm_strategy(AutoNameEnumBase):
        mean   = auto()
        max    = auto()
        kr     = auto()
        custom = auto()
    strategy to normalize the energy


    Returns
    ----------
    A function that takes HitCollection as input and returns HitCollection that containes
    only non NN hits of charge above threshold_charge with modified Ec and Z fields.
    """
    map_fname = os.path.expandvars(map_fname)
    maps      = cof.read_maps(map_fname)
    get_coef  = cof.apply_all_correction(maps, apply_temp = apply_temp, norm_strat = cof.norm_strategy.kr)
    if maps.t_evol is not None:
        time_to_Z = cof.get_df_to_z_converter(maps)
    else:
        time_to_Z = lambda x: x
    def threshold_and_correct_hits(hitc : evm.HitCollection) -> evm.HitCollection:
        t = hitc.time
        thr_hits = hif.threshold_hits(hitc.hits, threshold_charge     )
        mrg_hits = hif.merge_NN_hits ( thr_hits, same_peak = same_peak)
        X  = np.fromiter((h.X for h in mrg_hits), float)
        Y  = np.fromiter((h.Y for h in mrg_hits), float)
        Z  = np.fromiter((h.Z for h in mrg_hits), float)
        E  = np.fromiter((h.E for h in mrg_hits), float)
        Ec = E * get_coef(X,Y,Z,t)
        Zc = time_to_Z(Z)
        cor_hits = []
        for idx, hit in enumerate(mrg_hits):
            hit = evm.Hit(hit.npeak, evm.Cluster(hit.Q, xy(hit.X, hit.Y), hit.var, hit.nsipm),
                          Zc[idx], hit.E, xy(hit.Xpeak, hit.Ypeak), s2_energy_c = Ec[idx])
            cor_hits.append(hit)
        new_hitc      = evm.HitCollection(hitc.event, t)
        new_hitc.hits = cor_hits
        return new_hitc
    return threshold_and_correct_hits



@city
def esmeralda(files_in, file_out, compression, event_range, print_mod,
              detector_db, run_number,
              cor_hits_params = dict(),
              paolina_params  = dict()):
    """
    The city corrects Penthesilea hits energy and extracts topology information.
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

    cor_hits_params              : dict
        map_fname                : string (filepath)
            filename of the map
        threshold_charge_low     : float
            minimum pes for a lowTh hit
        threshold_charge_high    : float
            minimum pes for a highTh hit
        same_peak                : bool
            if True energy of NN hits is assigned only to the hits from the same peak
        apply_temp               : bool
            whether to apply temporal corrections
            must be set to False if no temporal correction dataframe exists in map file

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
    Penthesilea output
    ----------
    Output
    ----------
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


    cor_hits_params_   = {value : cor_hits_params.get(value) for value in ['map_fname', 'same_peak', 'apply_temp']}

    threshold_and_correct_hits_low  = fl.map(hits_threshold_and_corrector(threshold_charge=cor_hits_params['threshold_charge_low' ], **cor_hits_params_),
                                             args = 'hits',
                                             out  = 'cor_low_th_hits')

    threshold_and_correct_hits_high = fl.map(hits_threshold_and_corrector(threshold_charge=cor_hits_params['threshold_charge_high'], **cor_hits_params_),
                                             item = 'hits')

    filter_events_low_th            = fl.map(lambda x : len(x.hits) > 0,
                                             args = 'cor_low_th_hits',
                                             out  = 'low_th_hits_passed')

    hits_passed_low_th              = fl.count_filter(bool, args="low_th_hits_passed")

    event_count_in  = fl.spy_count()
    event_count_out = fl.count()

    with tb.open_file(file_out, "w", filters=tbl.filters(compression)) as h5out:

        # Define writers...
        write_event_info = fl.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))

        write_hits_low_th     = fl.sink(    hits_writer     (h5out, group_name='CHITS', table_name='lowTh'),
                                            args="cor_low_th_hits")
        write_hits_paolina    = fl.sink(    hits_writer     (h5out, group_name="CHITS", table_name="highTh" ),
                                            args="paolina_hits"   )
        write_hits_paolina_   = fl.branch(write_hits_paolina)

        write_low_th_filter   = fl.sink( event_filter_writer(h5out, "low_th_select"  )   , args=("event_number", "low_th_hits_passed" ))
        write_kdst_table      = fl.sink( kdst_from_df_writer(h5out)                      , args="kdst"               )


        compute_tracks = compute_and_write_tracks_info(paolina_params, h5out,
                                                       hit_type = evm.HitEnergy.Ec,
                                                       filter_hits_table_name = "high_th_select",
                                                       write_paolina_hits = write_hits_paolina_)

        evtnum_collect = collect()

        result = push(source = hits_and_kdst_from_files(files_in),
                      pipe   = pipe(fl.slice(*event_range, close_all=True)        ,
                                    print_every(print_mod)                        ,
                                    event_count_in        .spy                    ,
                                    fl.branch(fl.fork(write_kdst_table            ,
                                                      write_event_info          )),
                                    fl.branch("event_number", evtnum_collect.sink),
                                    fl.branch(threshold_and_correct_hits_low      ,
                                              filter_events_low_th                ,
                                              fl.branch(write_low_th_filter)      ,
                                              hits_passed_low_th.filter           ,
                                              write_hits_low_th                  ),
                                    threshold_and_correct_hits_high               ,
                                    compute_tracks                                ,
                                    "event_number"                                ,
                                    event_count_out       .sink                   ),
                      result = dict(events_in  =event_count_in .future,
                                    events_out =event_count_out.future,
                                    evtnum_list=evtnum_collect .future))

        if run_number <= 0:
            copy_mc_info(files_in, h5out, result.evtnum_list,
                         detector_db, run_number)

        return result
