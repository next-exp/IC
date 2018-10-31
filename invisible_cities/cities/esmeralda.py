"""
-----------------------------------------------------------------------
                              Esmeralda
-----------------------------------------------------------------------
This city is correcting hits and vixelizing them. The input is penthesilea output containing hits, kdst global information and mc info. The output contains tables:
- corrected hits
- voxels
- kdst global info 
- mc info


"""
from operator import attrgetter

import tables as tb

from .. dataflow            import dataflow as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe

from .  components import city
from .  components import print_every

from .. evm  import event_model as evm

from .. reco                  import tbl_functions        as tbl
from .. io.         hits_io import          hits_writer
from .. io.       mcinfo_io import       mc_info_writer
from .. io.run_and_event_io import run_and_event_writer
from .. io.       voxels_io import   true_voxels_writer
from .. io.         kdst_io import            kr_writer


def hits_and_kdst_from_files(paths : str )-> dict:
    """ source generator, yields hits and global info per event, and MC whole table  """
    pass

class HitsSelectorOutput:
    """
    Class devoted to hold the output of the HitsSelector.

    It contains:
        - passed  : a boolean flag indicating whether the event as
                    a whole has passed the filter.
        - hits   :  a boolean flag indicating whether the hit has
                    passed the filter.
    """
    pass

def hits_selector():
    def select_hits (hitc : evm.HitCollection)-> HitsSelectorOutput:
        """selects events and hits that passed the filter - probably all non NN hits """ 
        pass
    return select_hits

def NN_hits_merger():
    def merge_NN_hits (hitc:evm.HitCollection, hitc_pass:HitsSelectorOutput)-> evm.HitCollection:
        """ adds eneries of NN hits to surrounding reconstructed hits """
        pass
    return merge_NN_hits

def hits_corrector():
    def correct_hits  (hitc:evm.HitCollection, universal_cmap_interface)-> evm.HitCollection:
        """ corrects hits after merging of NN hits energies """
        pass
    return correct_hits

def hits_voxelizer():
    def voxelize_hits (hitc:evm.HitCollection)->evm.VoxelCollection:
        """Voxelize corrected hits"""
        pass
    return voxelize_hits


@city
def esmeralda(files_in, file_out, compression, event_range, print_mod, run_number,
              **args):
    
    select_hits   = fl.map(hits_selector(**locals()),
                           args = 'hits',
                           out  = 'hits_selector')
    
    hits_select   = fl.count_filter(attrgetter("passed"), args="hits_selector")
    
    merge_NN_hits = fl.map(NN_hits_merger(**locals()),
                           args = ('hits', 'hits_selector'),
                           out  = 'merged_hits')
    
    correct_hits  = fl.map(hits_corrector(**locals()),
                           args = 'merged_hits',
                           out  = 'corrected_hits')
    voxelize_hits = fl.map(hits_voxelizer(**locals()),
                           args = 'corrected_hits',
                           out  = 'voxels')
    
    event_count_in  = fl.spy_count()
    event_count_out = fl.spy_count()
    
    with tb.open_file(file_out, "w", filters = tbl.filters(compression)) as h5out:
        
        # Define writers...
        write_event_info = fl.sink(run_and_event_writer(h5out), args=("run_number", "event_number", "timestamp"))
        
        write_mc_        = mc_info_writer(h5out) if run_number <= 0 else (lambda *_: None)
        write_mc         = fl.sink(write_mc_, args=("mc", "event_number"))
        
        write_pointlike_event = fl.sink(kr_writer(h5out), args="pointlike_event")
        write_hits            = fl.sink(hits_writer(h5out), args="hits")
        write_voxels          = fl.sink(true_voxels_writer(h5out), args='voxels')
        
        return push(source = hits_and_kdst_from_files(files_in),
                    pipe   = pipe(
                        fl.slice(*event_range, close_all=True),
                        print_every(print_mod)                ,
                        event_count_in       .spy             ,
                        select_hits                           ,
                        hits_select          .filter          ,
                        event_count_out      .spy             ,
                        merge_NN_hits                         ,
                        correct_hits                          ,
                        voxelize_hits                         ,
                        fl.fork(write_mc                      ,
                                write_pointlike_event         ,
                                write_hits                    ,  
                                write_voxels                  ,
                                write_event_info             )),
                    result = dict(events_in  = event_count_in .future,
                                  events_out = event_count_out.future,
                                  selection  = hits_select    .future))
    
