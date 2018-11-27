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
import tables as tb
import numpy  as np
from functools   import partial

from typing      import Tuple
from typing      import Callable

from .. dataflow            import dataflow as fl
from .. dataflow.dataflow   import push
from .. dataflow.dataflow   import pipe

from .  components import city
from .  components import print_every

from .. evm  import event_model as evm

from .. reco                   import tbl_functions        as tbl
from .. reco.paolina_functions import voxelize_hits
from .. io.         hits_io import          hits_writer
from .. io.       mcinfo_io import       mc_info_writer
from .. io.run_and_event_io import run_and_event_writer
from .. io.       voxels_io import   true_voxels_writer
from .. io.         kdst_io import            kr_writer

from .. types.ic_types      import NN


def hits_and_kdst_from_files(paths : str )-> dict:
    """ source generator, yields hits and global info per event, and MC whole table  """
    pass

def merge_NN_hits(hitc : evm.HitCollection, same_peak : bool = True) -> Tuple[bool, evm.HitCollection]: 
    """ Returns a boolean flag if the event passed the filter and a modified HitCollection instance 
    by adding energies of NN hits to the closesthits such that the added energy is proportional to 
    the hit energy. If all the hits were NN the function returns bollean=False and  None.
    Note - the function will change El attribute in  hitc input object."""
    nn_hits     = [h for h in hitc.hits if h.Q==NN]
    non_nn_hits = [h for h in hitc.hits if h.Q!=NN]
    passed = len(non_nn_hits)>0
    if not passed:
        return passed, None
    for h in non_nn_hits:
        h.energy_l = h.E
    for nn_h in nn_hits:
        nn_h.energy_l = 0
        peak_num = nn_h.npeak
        if same_peak:
            hits_to_merge = [h for h in non_nn_hits if h.npeak==peak_num]
        else:
            hits_to_merge = non_nn_hits
        try:
            z_closest  = min(hits_to_merge , key=lambda h: np.abs(h.Z-nn_h.Z)).Z
        except ValueError:
            continue
        h_closest      = [h for h in hits_to_merge if h.Z==z_closest]
        h_closest_etot = sum([h.E for h in h_closest])
        for h in h_closest:
            h.energy_l += nn_h.E*(h.E/h_closest_etot)
    return passed,hitc

def NN_hits_merger(same_peak : bool = True) -> Callable:
    return partial(merge_NN_hits, same_peak=same_peak)

def hits_corrector():
    def correct_hits  (hitc:evm.HitCollection, universal_cmap_interface)-> evm.HitCollection:
        """ corrects hits after merging of NN hits energies """
        pass
    return correct_hits

def hits_voxelizer(voxel_size_X : float, voxel_size_Y : float, voxel_size_Z : float):
    return partial(voxelize_hits, voxel_dimensions = np.array([voxel_size_X, voxel_size_Y, voxel_size_Z]), strict_voxel_size = False)


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
    
