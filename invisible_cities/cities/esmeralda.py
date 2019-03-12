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
from functools   import partial
from typing      import Tuple
from typing      import Callable

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
from .. types.ic_types      import NN
from .. io.         hits_io import          hits_writer
from .. io.       mcinfo_io import       mc_info_writer
from .. io.run_and_event_io import run_and_event_writer
from .. io.          dst_io import _store_pandas_as_tables
def hits_threshold_and_corrector(map_fname: str, threshold_charge : float, same_peak : bool, apply_temp : bool) -> Callable:
    """Wrapper of correct_hits"""
    map_fname=os.path.expandvars(map_fname)
    maps=cof.read_maps(map_fname)
    get_coef=cof.apply_all_correction(maps, apply_temp = apply_temp)    
    def threshold_and_correct_hits(hitc : evm.HitCollection) -> evm.HitCollection:
        """ This function threshold the hits on the charge, redistribute the energy of NN hits to the surrouding ones and applies energy correction."""
        t = hitc.time
        cor_hitc = HitCollection(hitc.event, t)
        cor_hits = hif.threshold_hits(hitc.hits, threshold_charge)
        cor_hits = hif.merge_NN_hits(cor_hits, same_peak = same_peak)
        X  = np.array([h.X for h in cor_hits])
        Y  = np.array([h.Y for h in cor_hits])
        Z  = np.array([h.Z for h in cor_hits])
        E  = np.array([h.E for h in cor_hits])
        Ec = E * get_coef(X,Y,Z,t)
        Ec[np.isnan(Ec)] = NN
        for idx, hit in enumerate(cor_hits):
            hit.energy_c = Ec[idx]
        cor_hitc.hits = cor_hits
        return cor_hitc
    return threshold_and_correct_hits

def track_blob_info_extractor(vox_size, energy_type, energy_threshold, min_voxels, blob_radius, z_factor) -> Callable:
    """ Wrapper of extract_track_blob_info"""
    def extract_track_blob_info(hitc):
        """This function extract relevant info about the tracks and blobs, as well as assigning new field of energy, track_id etc to the HitCollection object (NOTE: we don't want to erase any hits, just redifine some attributes. If we need to cut away some hits to apply paolina functions, it has to be on the copy of the original hits)"""
        voxels     = plf.voxelize_hits(hitc.hits, vox_size, energy_type)
        mod_voxels = plf.drop_end_point_voxels(voxels, energy_threshold, min_voxels)
        tracks     = plf.make_track_graphs(mod_voxels)

        for t in tracks:
            min_z = min([h.Z for v in t.nodes() for h in v.hits])
            max_z = max([h.Z for v in t.nodes() for h in v.hits])

            for v in t.nodes():
                for h in v.hits:
                    setattr(h, energy_type.value,  getattr(h, energy_type.value) + getattr(h, energy_type.value)/(1 - z_factor * (max_z - min_z)))
                setattr(v, 'energy', getattr(v, 'energy') + getattr(v, 'energy')/(1 - z_factor * (max_z - min_z)))

        c_tracks = plf.make_track_graphs(mod_voxels)

        track_hits = []
        df = pd.DataFrame(columns=['trackID', 'energy', 'length', 'numb_of_voxels',
                                   'numb_of_hits', 'numb_of_tracks', 'x_min', 'y_min', 'z_min',
                                   'x_max', 'y_max', 'z_max', 'r_max', 'x_ave', 'y_ave', 'z_ave',
                                   'extreme1_x', 'extreme1_y', 'extreme1_z',
                                   'extreme2_x', 'extreme2_y', 'extreme2_z',
                                   'blob1_x', 'blob1_y', 'blob1_z',
                                   'blob2_x', 'blob2_y', 'blob2_z',
                                   'eblob1', 'eblob2', 'blob_overlap'])
        for c, t in enumerate(c_tracks, 0):
            tID = c
            energy = sum([vox.Ehits for vox in t.nodes()])
            length = plf.length(t)
            numb_of_hits = len([h for vox in t.nodes() for h in vox.hits])
            numb_of_voxels = len(t.nodes())
            numb_of_tracks = len(c_tracks)

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

            df.loc[c] = [tID, energy, length, numb_of_voxels, numb_of_hits, numb_of_tracks, min_x, min_y, min_z, max_x, max_y, max_z, max_r, ave_pos[0], ave_pos[1], ave_pos[2], extr1_pos[0], extr1_pos[1], extr1_pos[2], extr2_pos[0], extr2_pos[1], extr2_pos[2], blob_pos1[0], blob_pos1[1], blob_pos1[2], blob_pos2[0], blob_pos2[1], blob_pos2[2], e_blob1, e_blob2, overlap]

            for vox in t.nodes():
                for hit in vox.hits:
                    hit.track_id = tID
                    track_hits.append(hit)


        track_hitc = HitCollection(hitc.event, hitc.time)
        track_hitc.hits = track_hits

        return df, track_hitc

    return extract_track_blob_info


def final_summary_maker(**kargs)-> Callable:
    """I am not sure this is a new function or goes under extract_track_blob_info. To be discussed"""
    return partial(make_final_summary, **locals())


def make_final_summary(class_to_store_info_per_track, kdst_info_table,**kargs)-> class_to_store_event_summary:
    """I am not sure this is a new function or goes under extract_track_blob_info. To be discussed"""
    raise NotImplementedError



def track_writer(h5out, compression='ZLIB4', group_name='PAOLINA', table_name='Tracks', descriptive_string='Track information',str_col_length=32):
    def write_tracks(df):
        return _store_pandas_as_tables(h5out=h5out, df=df,compression = compression, group_name=group_name, table_name=table_name, descriptive_string=descriptive_string, str_col_length=str_col_length)
    return write_tracks


def summary_writer(h5out, compression='ZLIB4', group_name='PAOLINA', table_name='Summary', descriptive_string='Event summary information',str_col_length=32):
    def write_summary(df):
        return _store_pandas_as_tables(h5out=h5out, df=df,compression = compression, group_name=group_name, table_name=table_name, descriptive_string=descriptive_string, str_col_length=str_col_length)
    return write_summary


@city
def esmeralda(files_in, file_out, compression, event_range, print_mod, run_number, map_fname, **kargs):
    
    threshold_and_correct_hits_NN      = fl.map(hits_threshold_and_corrector(map_fname = map_fname,**locals()),
                                                args = 'hits',
                                                out  = 'NN_hits')

    threshold_and_correct_hits_paolina = fl.map(hits_threshold_and_corrector(map_fname = map_fname,**locals()),
                                                args = 'hits',
                                                out  = 'corrected_hits')

    extract_track_blob_info = fl.map(track_blob_info_extractor(vox_size, energy_type, energy_threshold, min_voxels, blob_radius, z_factor),
                                     args = 'corrected_hits',
                                     out  = ('paolina_hits', 'topology_info'))

    make_final_summary      = fl.map(final_summary_maker(**locals()),
                                     args = 'topology_info',
                                     out  = 'event_info')

    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()

    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:

        # Define writers...
        write_event_info = fl.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))
        write_mc_        = mc_info_writer(h5out) if run_number <= 0 else (lambda *_: None)

        write_mc           = fl.sink(             write_mc_, args = ("mc", "event_number"   ))
        write_hits_NN      = fl.sink(    hits_writer(h5out), args =  "NN_hits"               )
        write_hits_paolina = fl.sink(    hits_writer(h5out, group_name = 'PAOLINA'), args =  "paolina_hits"          )
        write_summary      = fl.sink( summary_writer(h5out), args =  "event_info"            )

        return push(source = hits_and_kdst_from_files(files_in),
                    pipe   = pipe(
                        fl.slice(*event_range, close_all=True)     ,
                        print_every(print_mod)                     ,
                        event_count_in       .spy                  ,
                        fl.branch(threshold_and_correct_hits_NN    ,
                                  write_hits_NN)                   ,
                        threshold_and_correct_hits_paolina         ,
                        extract_track_blob_info                    ,
                        fl.fork(write_mc                           ,
                                write_hits_paolina                 ,
                                (make_final_summary, write_summary),
                                write_event_info)),
                    result = dict(events_in  = event_count_in .future,
                                  events_out = event_count_out.future))
