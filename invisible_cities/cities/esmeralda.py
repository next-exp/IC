"""
-----------------------------------------------------------------------
                              Esmeralda
-----------------------------------------------------------------------
From Spanish esmeralda (“emerald”), as first used in the novel Notre-Dame de Paris (1831) by Victor Hugo.
This city is correcting hits and voxelizing them. The input is penthesilea output containing hits, kdst global information and mc info. The output contains tables:
- corrected hits
- summary of topology analysis
- mc info

"""
import os
import tables as tb
import numpy  as np
import pandas as pd
import warnings

from functools   import partial
from typing      import Tuple
from typing      import Callable
from typing      import Optional

from .. reco                import tbl_functions        as tbl
from .. reco                import paolina_functions    as plf
from .. reco                import hits_functions       as hif
from .. reco                import corrections_new      as cof
from .. dataflow            import dataflow as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe

from .  components import city
from .  components import print_every
from .  components import hits_and_kdst_from_files
from .. evm  import event_model as evm
from .. types.      ic_types import NN
from .. types.      ic_types import xy
from .. io.         hits_io  import          hits_writer
from .. io.       mcinfo_io  import       mc_info_writer
from .. io.run_and_event_io  import run_and_event_writer
from .. io. event_filter_io  import  event_filter_writer
from .. io.          dst_io  import _store_pandas_as_tables


def hits_threshold_and_corrector(map_fname: str, threshold_charge : float, same_peak : bool, apply_temp : bool) -> Callable:
    """Wrapper of correct_hits"""
    map_fname=os.path.expandvars(map_fname)
    maps=cof.read_maps(map_fname)
    get_coef=cof.apply_all_correction(maps, apply_temp = apply_temp)
    def threshold_and_correct_hits(hitc : evm.HitCollection) -> evm.HitCollection:
        """ This function threshold the hits on the charge, redistribute the energy of NN hits to the surrouding ones and applies energy correction."""
        t = hitc.time
        thr_hits = hif.threshold_hits(hitc.hits, threshold_charge     )
        mrg_hits = hif.merge_NN_hits ( thr_hits, same_peak = same_peak)
        if len(mrg_hits) == 0:
            return None
        X  = np.array([h.X for h in mrg_hits])
        Y  = np.array([h.Y for h in mrg_hits])
        Z  = np.array([h.Z for h in mrg_hits])
        E  = np.array([h.E for h in mrg_hits])
        Ec = E * get_coef(X,Y,Z,t)
        #Ec[np.isnan(Ec)] = NN
        cor_hits = []
        for idx, hit in enumerate(mrg_hits):
            hit = evm.Hit(hit.npeak, evm.Cluster(hit.Q, xy(hit.X, hit.Y), hit.var, hit.nsipm), hit.Z, hit.E, xy(hit.Xpeak, hit.Ypeak), s2_energy_c = Ec[idx])
            cor_hits.append(hit)
        new_hitc       = evm.HitCollection(hitc.event, t)
        new_hitc.hits = cor_hits
        return new_hitc
    return threshold_and_correct_hits


def events_filter(allow_nans : bool) -> Callable:
    def filter_events(hitc : Optional[evm.HitCollection]) -> bool:
        if hitc == None:
            return False
        elif allow_nans == False:
            nans = np.isnan([h.Ec for h in hitc.hits])
            return not(any(nans))
        else:
            return True
    return filter_events


def track_blob_info_extractor(vox_size, energy_type, energy_threshold, min_voxels, blob_radius) -> Callable:
    """ Wrapper of extract_track_blob_info"""
    def extract_track_blob_info(hitc):
        """This function extract relevant info about the tracks and blobs, as well as assigning new field of energy, track_id etc to the HitCollection object (NOTE: we don't want to erase any hits, just redifine some attributes. If we need to cut away some hits to apply paolina functions, it has to be on the copy of the original hits)"""
        voxels     = plf.voxelize_hits(hitc.hits, vox_size, energy_type)
        mod_voxels = plf.drop_end_point_voxels(voxels, energy_threshold, min_voxels)
        tracks     = plf.make_track_graphs(mod_voxels)

        vox_size_x = voxels[0].size[0]
        vox_size_y = voxels[0].size[1]
        vox_size_z = voxels[0].size[2]

        for t in tracks:
            min_z = min([h.Z for v in t.nodes() for h in v.hits])
            max_z = max([h.Z for v in t.nodes() for h in v.hits])


        track_hits = []
        df = pd.DataFrame(columns=['event', 'trackID', 'energy', 'length', 'numb_of_voxels',
                                   'numb_of_hits', 'numb_of_tracks', 'x_min', 'y_min', 'z_min',
                                   'x_max', 'y_max', 'z_max', 'r_max', 'x_ave', 'y_ave', 'z_ave',
                                   'extreme1_x', 'extreme1_y', 'extreme1_z',
                                   'extreme2_x', 'extreme2_y', 'extreme2_z',
                                   'blob1_x', 'blob1_y', 'blob1_z',
                                   'blob2_x', 'blob2_y', 'blob2_z',
                                   'eblob1', 'eblob2', 'blob_overlap',
                                   'vox_size_x', 'vox_size_y', 'vox_size_z'])
        for c, t in enumerate(tracks, 0):
            tID = c
            energy = sum([vox.Ehits for vox in t.nodes()])
            length = plf.length(t)
            numb_of_hits = len([h for vox in t.nodes() for h in vox.hits])
            numb_of_voxels = len(t.nodes())
            numb_of_tracks = len(tracks   )

            min_x = min([h.X for v in t.nodes() for h in v.hits])
            max_x = max([h.X for v in t.nodes() for h in v.hits])
            min_y = min([h.Y for v in t.nodes() for h in v.hits])
            max_y = max([h.Y for v in t.nodes() for h in v.hits])
            min_z = min([h.Z for v in t.nodes() for h in v.hits])
            max_z = max([h.Z for v in t.nodes() for h in v.hits])
            max_r = max([np.sqrt(h.X*h.X + h.Y*h.Y) for v in t.nodes() for h in v.hits])

            pos = [h.pos for v in t.nodes() for h in v.hits]
            e   = [getattr(h, energy_type.value) for v in t.nodes() for h in v.hits]
            ave_pos = np.average(pos, weights=e, axis=0)

            extr1, extr2 = plf.find_extrema(t)
            extr1_pos = extr1.XYZ
            extr2_pos = extr2.XYZ

            blob_pos1, blob_pos2 = plf.blob_centres(t, blob_radius)

            e_blob1, e_blob2, hits_blob1, hits_blob2 = plf.blob_energies_and_hits(t, blob_radius)
            overlap = False
            if len(set(hits_blob1).intersection(hits_blob2)) > 0:
                overlap = True
            list_of_vars = [hitc.event, tID, energy, length, numb_of_voxels, numb_of_hits, numb_of_tracks, min_x, min_y, min_z, max_x, max_y, max_z, max_r, ave_pos[0], ave_pos[1], ave_pos[2], extr1_pos[0], extr1_pos[1], extr1_pos[2], extr2_pos[0], extr2_pos[1], extr2_pos[2], blob_pos1[0], blob_pos1[1], blob_pos1[2], blob_pos2[0], blob_pos2[1], blob_pos2[2], e_blob1, e_blob2, overlap, vox_size_x, vox_size_y, vox_size_z]

            df.loc[c] = list_of_vars
            try:
                types_dict
            except NameError:
                types_dict = dict(zip(df.columns, [type(x) for x in list_of_vars]))

            for vox in t.nodes():
                for hit in vox.hits:
                    hit.track_id = tID
                    track_hits.append(hit)


        track_hitc = evm.HitCollection(hitc.event, hitc.time)
        track_hitc.hits = track_hits
        #change dtype of columns to match type of variables
        df = df.apply(lambda x : x.astype(types_dict[x.name]))

        return df, track_hitc

    return extract_track_blob_info


def make_event_summary(event_number, timestamp, topology_info, paolina_hits, kdst) -> pd.DataFrame:
    """Compute the quantities to be placed in the final event summary"""
    es = pd.DataFrame(columns=['event', 'time', 'S1e', 'S1t',
                               'nS2', 'ntrks', 'nhits', 'S2e0', 'S2ec',
                               'S2q0', 'S2qc', 'x_avg', 'y_avg', 'z_avg',
                               'r_avg', 'x_min', 'y_min', 'z_min', 'r_min',
                               'x_max', 'y_max', 'z_max', 'r_max'])

    if(len(kdst.s1_peak.unique()) != 1):
        warnings.warn("Number of recorded S1 energies differs from 1 in event {}.  Choosing first S1".format(event_number), UserWarning)

    S1e = kdst.S1e.values[0]
    S1t = kdst.S1t.values[0]

    nS2 = kdst.nS2.values[0]

    ntrks = len(topology_info.index)
    nhits = len(paolina_hits.hits)

    S2e0 = np.sum(kdst.S2e.values)
    S2ec = sum([h.E for h in paolina_hits.hits])

    S2q0 = np.sum(kdst.S2q.values)
    S2qc = sum([h.Q for h in paolina_hits.hits])

    x_avg = sum([h.X*h.E for h in paolina_hits.hits])
    y_avg = sum([h.Y*h.E for h in paolina_hits.hits])
    z_avg = sum([h.Z*h.E for h in paolina_hits.hits])
    r_avg = sum([(h.X**2 + h.Y**2)**0.5*h.E for h in paolina_hits.hits])
    if(S2ec > 0):
        x_avg /= S2ec
        y_avg /= S2ec
        z_avg /= S2ec
        r_avg /= S2ec

    x_min = min([h.X for h in paolina_hits.hits])
    y_min = min([h.Y for h in paolina_hits.hits])
    z_min = min([h.Z for h in paolina_hits.hits])
    r_min = min([(h.X**2 + h.Y**2)**0.5 for h in paolina_hits.hits])

    x_max = max([h.X for h in paolina_hits.hits])
    y_max = max([h.Y for h in paolina_hits.hits])
    z_max = max([h.Z for h in paolina_hits.hits])
    r_max = max([(h.X**2 + h.Y**2)**0.5 for h in paolina_hits.hits])

    list_of_vars  = [event_number, int(timestamp), S1e, S1t, nS2, ntrks, nhits, S2e0,
                     S2ec, S2q0, S2qc, x_avg, y_avg, z_avg, r_avg, x_min, y_min,
                     z_min, r_min, x_max, y_max, z_max, r_max]

    es.loc[0] = list_of_vars
    #change dtype of columns to match type of variables
    types_dict = dict(zip(es.columns, [type(x) for x in list_of_vars]))
    es = es.apply(lambda x : x.astype(types_dict[x.name]))

    return es


def track_writer(h5out, compression='ZLIB4', group_name='PAOLINA', table_name='Tracks', descriptive_string='Track information',str_col_length=32):
    def write_tracks(df):
        return _store_pandas_as_tables(h5out=h5out, df=df,compression = compression, group_name=group_name, table_name=table_name, descriptive_string=descriptive_string, str_col_length=str_col_length)
    return write_tracks


def summary_writer(h5out, compression='ZLIB4', group_name='PAOLINA', table_name='Summary', descriptive_string='Event summary information',str_col_length=32):
    def write_summary(df):
        return _store_pandas_as_tables(h5out=h5out, df=df,compression = compression, group_name=group_name, table_name=table_name, descriptive_string=descriptive_string, str_col_length=str_col_length)
    return write_summary


@city
def esmeralda(files_in, file_out, compression, event_range, print_mod, run_number,
              cor_hits_params_NN = dict(),
              cor_hits_params_PL = dict(),
              paolina_params     = dict()):
    # cor_hits_params_NN   are map_fname, threshold_charge, same_peak, apply_temp
    # cor_hits_params_PL   are map_fname, threshold_charge, same_peak, apply_temp
    # paolina_params       are vox_size,  energy_type, energy_threshold, min_voxels, blob_radius, z_factor
    # energy_type parameter in paolina_params has to be translated to enum class
    if   paolina_params['energy_type'] == 'corrected'   : paolina_params['energy_type'] = evm.HitEnergy.Ec
    elif paolina_params['energy_type'] == 'uncorrected' : paolina_params['energy_type'] = evm.HitEnergy.E
    else                                                : raise ValueError(f"Unrecognized processing mode: {paolina_params['energy_type']}")


    threshold_and_correct_hits_NN      = fl.map(hits_threshold_and_corrector(**cor_hits_params_NN),
                                                args = 'hits',
                                                out  = 'NN_hits')

    threshold_and_correct_hits_paolina = fl.map(hits_threshold_and_corrector(**cor_hits_params_PL),
                                                args = 'hits',
                                                out  = 'corrected_hits')

    filter_events_NN      = fl.map(events_filter(allow_nans = True),
                                   args = 'NN_hits',
                                   out  = 'NN_hits_passed')

    filter_events_paolina = fl.map(events_filter(allow_nans = False),
                              args = 'corrected_hits',
                              out  = 'paolina_hits_passed')

    hits_passed_NN        = fl.count_filter(bool, args =      "NN_hits_passed")
    hits_passed_paolina   = fl.count_filter(bool, args = "paolina_hits_passed")

    extract_track_blob_info = fl.map(track_blob_info_extractor(**paolina_params),
                                     args = 'corrected_hits',
                                     out  = ('topology_info', 'paolina_hits'))

    make_final_summary      = fl.map(make_event_summary,
                                     args = ('event_number', 'timestamp', 'topology_info', 'paolina_hits', 'kdst'),
                                     out  = 'event_info')

    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        # Define writers...
        write_event_info = fl.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))
        write_mc_        = mc_info_writer(h5out) if run_number <= 0 else (lambda *_: None)

        write_mc             = fl.sink(    write_mc_                                      , args = ("mc", "event_number"))
        write_hits_NN        = fl.sink(    hits_writer     (h5out)                        , args =  "NN_hits"            )
        write_hits_paolina   = fl.sink(    hits_writer     (h5out, group_name = 'PAOLINA'), args =  "paolina_hits"       )
        write_tracks         = fl.sink(   track_writer     (h5out=h5out)                  , args =  "topology_info"      )
        write_summary        = fl.sink( summary_writer     (h5out=h5out)                  , args =  "event_info"         )
        write_paolina_filter = fl.sink( event_filter_writer(h5out, "paolina_select")      , args = ("event_number", "paolina_hits_passed"))
        write_NN_filter      = fl.sink( event_filter_writer(h5out,      "NN_select")      , args = ("event_number",      "NN_hits_passed"))

        return push(source = hits_and_kdst_from_files(files_in),
                    pipe   = pipe(
                        fl.slice(*event_range, close_all=True)     ,
                        print_every(print_mod)                     ,
                        event_count_in           .spy              ,
                        fl.branch(threshold_and_correct_hits_NN    ,
                                  filter_events_NN                 ,
                                  fl.branch(write_NN_filter)       ,
                                  hits_passed_NN .filter           ,
                                  write_hits_NN)                   ,
                        threshold_and_correct_hits_paolina         ,
                        filter_events_paolina                      ,
                        fl.branch(write_paolina_filter)            ,
                        hits_passed_paolina      .filter           ,
                        event_count_out          .spy              ,
                        extract_track_blob_info                    ,
                        fl.fork(write_mc                           ,
                                write_hits_paolina                 ,
                                write_tracks                       ,
                                (make_final_summary, write_summary),
                                write_event_info))                 ,
                    result = dict(events_in  = event_count_in .future,
                                  events_out = event_count_out.future))
