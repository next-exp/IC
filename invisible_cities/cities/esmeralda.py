"""
-----------------------------------------------------------------------
                              Esmeralda
-----------------------------------------------------------------------
This city is correcting hits and vixelizing them. The input is penthesilea output containing hits, kdst global information and mc info. The output contains tables:
- corrected hits
- summary of topology analysis
- mc info

"""
import tables as tb
import numpy  as np
from functools   import partial
from typing      import Tuple
from typing      import Callable

from .. reco                import tbl_functions        as tbl
from .. reco                import paolina_functions    as plf
from .. dataflow            import dataflow as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe

from .  components import city
from .  components import print_every
from .  components import hits_and_kdst_from_files
from .. evm  import event_model as evm

from .. io.         hits_io import          hits_writer
from .. io.       mcinfo_io import       mc_info_writer
from .. io.run_and_event_io import run_and_event_writer

def hits_threshold_and_corrector(map_fname: str, **kargs) -> Callable:
    """Wrapper of correct_hits"""
    return partial(threshold_and_correct_hits(**locals()))

def track_blob_info_extractor(vox_size, energy_threshold, min_voxels) -> Callable:
    """ Wrapper of extract_track_blob_info"""
    def extract_track_blob_info(hitc):
        """This function extract relevant info about the tracks and blobs, as well as assigning new field of energy, track_id etc to the HitCollection object (NOTE: we don't want to erase any hits, just redifine some attributes. If we need to cut away some hits to apply paolina functions, it has to be on the copy of the original hits)"""
        voxels     = plf.voxelize_hits(hitc, vox_size)
        mod_voxels = plf.drop_end_point_voxels(voxels, energy_threshold, min_voxels)
        tracks     = plf.make_track_graphs(mod_voxels)

    return extract_track_blob_info


def final_summary_maker(**kargs)-> Callable:
    """I am not sure this is a new function or goes under extract_track_blob_info. To be discussed"""
    return partial(make_final_summary, **locals())

#Function to define
def threshold_and_correct_hits(hitc : evm.HitCollection, **kargs) -> evm.HitCollection:
    """ This function threshold the hits on the charge, redistribute the energy of NN hits to the surrpouding ones and applies energy correction."""
    raise NotImplementedError

class class_to_store_info_per_track:
    pass

class class_to_store_event_summary:
    pass


def extract_track_blob_info(hitc : evm.HitCollection, **kargs)-> Tuple(evm.HitCollection, class_to_store_info_per_track):
    """This function extract relevant info about the tracks and blobs, as well as assigning new field of energy, track_id etc to the HitCollection object (NOTE: we don't want to erase any hits, just redifine some attributes. If we need to cut away some hits to apply paolina functions, it has to be on the copy of the original hits)"""
    raise NotImplementedError


def make_final_summary(class_to_store_info_per_track, kdst_info_table,**kargs)-> class_to_store_event_summary:
    """I am not sure this is a new function or goes under extract_track_blob_info. To be discussed"""
    raise NotImplementedError


def summary_writer(hdf5_file, *, compression='ZLIB4'):
    def write_summary(summary_info : class_to_store_event_summary):
        raise NotImplementedError
    return write_summary

@city
def esmeralda(files_in, file_out, compression, event_range, print_mod, run_number, map_fname, **kargs):
    
    threshold_and_correct_hits_NN      = fl.map(hits_threshold_and_corrector(map_fname = map_fname,**locals()),
                                                args = 'hits',
                                                out  = 'NN_hits')

    threshold_and_correct_hits_paolina = fl.map(hits_threshold_and_corrector(map_fname = map_fname,**locals()),
                                                args = 'hits',
                                                out  = 'corrected_hits')

    extract_track_blob_info = fl.map(track_blob_info_extractor(vox_size, energy_threshold, min_voxels),
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
        write_hits_paolina = fl.sink(    hits_writer(h5out), args =  "paolina_hits"          )
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
